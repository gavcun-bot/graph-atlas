"""Exercise the shared graph contract from demo generation through MCP."""

from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from atlas.__main__ import graph_rows, import_graph
from atlas.mcp import get_neighbors, open_graph, search_nodes
from web.build_demo import build_graph


class GraphAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name)
        cls.graph = build_graph()
        cls.input = path / "demo.json"
        cls.input.write_text(json.dumps(cls.graph), encoding="utf-8")
        cls.database = path / "demo.sqlite"
        import_graph(cls.input, cls.database)

    def test_search_and_relationship_evidence(self):
        with closing(open_graph(self.database)) as connection:
            result = search_nodes(connection, "Ocean Signal Observatory", limit=1)
            node = result["items"][0]
            self.assertEqual(node["id"], "station:ocean-01")
            rows = get_neighbors(connection, node["id"], limit=2)
            self.assertEqual(len(rows["items"]), 2)
            self.assertEqual(rows["next_offset"], 2)
            for row in rows["items"]:
                self.assertIn(row["direction"], {"outgoing", "incoming"})
                self.assertTrue(row["evidence"])
            more = get_neighbors(connection, node["id"], limit=2, offset=2)
            self.assertNotEqual(rows["items"], more["items"])

    def test_import_rejects_invalid_endpoints_and_existing_destination(self):
        graph = {
            "nodes": self.graph["nodes"][:1],
            "edges": [[0, 99, "mentions", "Example"]],
        }
        with self.assertRaisesRegex(ValueError, "missing node"):
            graph_rows(graph)
        with self.assertRaises(FileExistsError):
            import_graph(self.input, self.database)

    def test_mcp_stdio_discovery_and_read(self):
        messages = [
            {"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
            {"method": "notifications/initialized"},
            {"id": 2, "method": "tools/list"},
            {"id": 3, "method": "tools/call", "params": {
                "name": "get_node", "arguments": {"id": "station:ocean-01"},
            }},
        ]
        client = subprocess.run(
            [sys.executable, "-m", "atlas.mcp", "--database", str(self.database)],
            input="\n".join(json.dumps({"jsonrpc": "2.0", **m}) for m in messages) + "\n",
            text=True, capture_output=True, check=True, timeout=10,
        )
        responses = [json.loads(line) for line in client.stdout.splitlines()]
        self.assertEqual([response["id"] for response in responses], [1, 2, 3])
        self.assertEqual(len(responses[1]["result"]["tools"]), 3)
        self.assertEqual(
            json.loads(responses[2]["result"]["content"][0]["text"])["label"],
            "Ocean Signal Observatory",
        )


if __name__ == "__main__":
    unittest.main()
