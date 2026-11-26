"""Read an evidence graph over MCP stdio. Requires Python 3.9+.

Usage: python -m atlas.mcp --database demo.sqlite
The SQLite schema is documented in README.md. All tool arguments and returned
records pass over the standard streams of the client that launches the server.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
TEXT = {"type": "string", "minLength": 1, "maxLength": 512}
PAGE = {
    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
    "offset": {"type": "integer", "minimum": 0, "default": 0},
}
NODE_COLUMNS = "id, label, type, source, detail"


def tool(name, description, required, properties):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object", "properties": properties,
            "required": required, "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True, "destructiveHint": False,
            "idempotentHint": True, "openWorldHint": False,
        },
    }


TOOLS = {
    entry["name"]: entry for entry in (
        tool("search_nodes", "Find entities by ID, label or description. Exact ID and label matches rank first.",
             ["query"], {"query": TEXT, "type": TEXT, **PAGE}),
        tool("get_node", "Read an entity, its provenance and description using its exact ID.",
             ["id"], {"id": TEXT}),
        tool("get_neighbors", "List incident relationships with direction, evidence and the other endpoint.",
             ["id"], {"id": TEXT, "relation": TEXT, **PAGE}),
    )
}


def open_graph(path):
    database = sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA query_only = ON")
    database.execute("PRAGMA trusted_schema = OFF")
    database.execute("SELECT id, label, type, source, detail FROM nodes LIMIT 0")
    database.execute("SELECT source, target, type, evidence FROM edges LIMIT 0")
    return database


def page_rows(cursor, limit, offset):
    rows = cursor.fetchall()
    return {
        "items": [dict(row) for row in rows[:limit]],
        "next_offset": offset + limit if len(rows) > limit else None,
    }


def search_nodes(database, query, type=None, limit=20, offset=0):
    """Search graph nodes with bound values and stable pagination."""
    cursor = database.execute(
        f"""SELECT {NODE_COLUMNS} FROM nodes
            WHERE (? IS NULL OR type = ?)
              AND (instr(lower(id), lower(?)) > 0 OR instr(lower(label), lower(?)) > 0
                   OR instr(lower(detail), lower(?)) > 0)
            ORDER BY CASE WHEN lower(id) = lower(?) OR lower(label) = lower(?) THEN 0
                          WHEN instr(lower(label), lower(?)) = 1 THEN 1
                          WHEN instr(lower(label), lower(?)) > 0 THEN 2 ELSE 3 END,
                     label COLLATE NOCASE, id
            LIMIT ? OFFSET ?""",
        (type, type, query, query, query, query, query, query, query, limit + 1, offset),
    )
    return page_rows(cursor, limit, offset)


def get_node(database, id):
    """Return one source-backed entity by its exact identifier."""
    row = database.execute(
        f"SELECT {NODE_COLUMNS} FROM nodes WHERE id = ?", (id,)
    ).fetchone()
    if row is None:
        raise ValueError("Unknown entity ID")
    return dict(row)


def get_neighbors(database, id, relation=None, limit=20, offset=0):
    """Read both incoming and outgoing edge evidence around one node."""
    get_node(database, id)
    cursor = database.execute(
        """SELECT e.source, e.target, e.type AS relation, e.evidence,
                  CASE WHEN e.source = :id THEN 'outgoing' ELSE 'incoming' END AS direction,
                  n.id AS neighbor_id, n.label AS neighbor_label, n.type AS neighbor_type
           FROM edges AS e
           JOIN nodes AS n ON n.id = CASE WHEN e.source = :id THEN e.target ELSE e.source END
           WHERE (e.source = :id OR e.target = :id)
             AND (:relation IS NULL OR e.type = :relation)
           ORDER BY e.type, neighbor_label COLLATE NOCASE, n.id, e.source, e.target, e.evidence
           LIMIT :fetch OFFSET :offset""",
        {"id": id, "relation": relation, "fetch": limit + 1, "offset": offset},
    )
    return page_rows(cursor, limit, offset)


FUNCTIONS = {f.__name__: f for f in (search_nodes, get_node, get_neighbors)}


def validate_arguments(schema, values):
    if not isinstance(values, dict):
        raise ValueError("Tool arguments must be an object")
    if set(schema["required"]) - values.keys():
        raise ValueError("A required argument is missing")
    for key, value in values.items():
        rule = schema["properties"].get(key)
        if not rule:
            raise ValueError("Unknown argument")
        if rule["type"] == "string":
            if not isinstance(value, str) or not value.strip() or len(value) > rule["maxLength"]:
                raise ValueError(f"Invalid string for {key}")
        elif type(value) is not int or not rule["minimum"] <= value <= rule.get("maximum", 2**63 - 1):
            raise ValueError(f"Invalid integer for {key}")


def error_response(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_message(database, message, state):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return error_response(None, -32600, "Invalid JSON-RPC request")
    method = message["method"]
    if "id" not in message:
        if method == "notifications/initialized" and state["phase"] == "initializing":
            state["phase"] = "ready"
        return None
    request_id = message["id"]
    if type(request_id) not in (str, int):
        return error_response(None, -32600, "Invalid request ID")
    params = message.get("params", {})
    if not isinstance(params, dict):
        return error_response(request_id, -32602, "Invalid request parameters")
    if method == "initialize":
        if state["phase"] != "new":
            return error_response(request_id, -32600, "Already initialized")
        requested = params.get("protocolVersion")
        if not isinstance(requested, str):
            return error_response(request_id, -32602, "Missing protocolVersion")
        result = {
            "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "graph-atlas", "version": "1.0.0"},
        }
        state["phase"] = "initializing"
    elif method == "ping":
        result = {}
    elif state["phase"] != "ready":
        return error_response(request_id, -32002, "Initialization required")
    elif method == "tools/list":
        result = {"tools": list(TOOLS.values())}
    elif method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str) or name not in TOOLS:
            return error_response(request_id, -32602, "Unknown tool")
        values = params.get("arguments", {})
        try:
            validate_arguments(TOOLS[name]["inputSchema"], values)
            record = FUNCTIONS[name](database, **values)
            result = {"content": [{"type": "text", "text": json.dumps(record, ensure_ascii=False)}]}
        except ValueError as exc:
            result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        except sqlite3.Error:
            result = {"isError": True, "content": [{"type": "text", "text": "Database read failed"}]}
    else:
        return error_response(request_id, -32601, "Unknown method")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(database):
    state = {"phase": "new"}
    for line in sys.stdin.buffer:
        try:
            response = handle_message(database, json.loads(line), state)
        except (ValueError, UnicodeError, RecursionError):
            response = error_response(None, -32700, "Invalid JSON")
        if response is not None:
            encoded = json.dumps(response, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            sys.stdout.buffer.write(encoded + b"\n")
            sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    try:
        database = open_graph(args.database)
    except (OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Cannot open the graph database: {exc}\n")
    try:
        serve(database)
    finally:
        database.close()


if __name__ == "__main__":
    main()
