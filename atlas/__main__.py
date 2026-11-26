"""Load a JSON evidence graph into SQLite or run a read-only query."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from .mcp import FUNCTIONS, TOOLS, open_graph, validate_arguments


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE nodes (
    id TEXT PRIMARY KEY NOT NULL,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    detail TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE edges (
    source TEXT NOT NULL REFERENCES nodes(id),
    target TEXT NOT NULL REFERENCES nodes(id),
    type TEXT NOT NULL,
    evidence TEXT NOT NULL
);
CREATE INDEX edges_by_source ON edges (source, type);
CREATE INDEX edges_by_target ON edges (target, type);
CREATE INDEX nodes_by_type ON nodes (type);
"""


def text_field(value, field, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be text" + ("" if allow_empty else " and cannot be blank"))
    return value


def graph_rows(graph):
    """Resolve the viewer's node offsets to stable IDs after validating the payload."""
    if not isinstance(graph, dict):
        raise ValueError("Expected a JSON object with nodes and edges")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Both nodes and edges must be JSON arrays")

    node_rows = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a JSON object")
        fields = ("id", "label", "type", "source", "detail")
        row = tuple(
            text_field(node.get(key, "" if key == "detail" else None), f"Node {index} {key}",
                       allow_empty=key == "detail")
            for key in fields
        )
        node_rows.append(row)
    if len({row[0] for row in node_rows}) != len(node_rows):
        raise ValueError("Node IDs must be unique")

    edge_rows = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 4:
            raise ValueError(f"Edge {index} must have two offsets, a relationship and evidence")
        a, b, relationship, evidence = edge
        if any(type(offset) is not int or not 0 <= offset < len(nodes) for offset in (a, b)):
            raise ValueError(f"Edge {index} refers to a missing node")
        edge_rows.append((
            node_rows[a][0], node_rows[b][0],
            text_field(relationship, f"Edge {index} relationship"),
            text_field(evidence, f"Edge {index} evidence"),
        ))
    return node_rows, edge_rows


def import_graph(source, destination):
    """Create a complete new database without overwriting an existing snapshot."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    nodes, edges = graph_rows(json.loads(source.read_text(encoding="utf-8")))
    with TemporaryDirectory(prefix=".atlas-import-", dir=destination.parent) as temporary:
        database_path = Path(temporary) / "graph.sqlite"
        with closing(sqlite3.connect(database_path)) as db:
            db.executescript(SCHEMA)
            with db:
                db.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?)", nodes)
                db.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", edges)
        os.link(database_path, destination)
    return {"nodes": len(nodes), "edges": len(edges), "database": destination.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    importer = commands.add_parser("import", aliases=["import-demo"], help="Create a database from graph JSON")
    importer.add_argument("source", type=Path)
    importer.add_argument("destination", type=Path)

    for command, tool_name, argument in (
        ("search", "search_nodes", "query"),
        ("node", "get_node", "id"),
        ("neighbors", "get_neighbors", "id"),
    ):
        subparser = commands.add_parser(command, help=TOOLS[tool_name]["description"])
        subparser.set_defaults(tool_name=tool_name)
        subparser.add_argument("database", type=Path)
        subparser.add_argument(argument)
        if command in {"search", "neighbors"}:
            subparser.add_argument("--limit", type=int)
            subparser.add_argument("--offset", type=int)
            subparser.add_argument("--type" if command == "search" else "--relation")

    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command in {"import", "import-demo"}:
            result = import_graph(args["source"], args["destination"])
        else:
            tool_name = args.pop("tool_name")
            database_path = args.pop("database")
            values = {key: value for key, value in args.items() if value is not None}
            validate_arguments(TOOLS[tool_name]["inputSchema"], values)
            with closing(open_graph(database_path)) as db:
                result = FUNCTIONS[tool_name](db, **values)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
