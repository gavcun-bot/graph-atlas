"""Detect communities and lay out a graph using Graphviz's sfdp engine."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

PALETTE = (
    "#8251a6", "#417bc3", "#238a82", "#679536", "#cd6d35", "#ba5075",
    "#3287a8", "#866440", "#ad4349", "#42639c", "#8e7540", "#4f8762",
)
PLACEHOLDER = "#75838e"


def index_adjacency(graph):
    """Count non-loop edges as interaction weights without changing their records."""
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Graph must have nodes and edges arrays")

    ids = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise ValueError("Every node must have a nonempty string ID")
        if node["id"] in ids:
            raise ValueError(f"Duplicate node ID: {node['id']}")
        ids.add(node["id"])

    adjacency = [Counter() for _ in nodes]
    degree = [0 for _ in nodes]
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 4:
            raise ValueError("Every edge must contain source, target, relationship and evidence")
        source, target, _, _ = edge
        if any(type(index) is not int or not 0 <= index < len(nodes) for index in (source, target)):
            raise ValueError("Edge endpoints must be valid integer node indices")
        degree[source] += 1
        if source != target:
            degree[target] += 1
            adjacency[source][target] += 1
            adjacency[target][source] += 1
    return adjacency, degree


def modularity_communities(adjacency, resolution, rng):
    """Repeatedly move vertices to improve modularity, then contract communities."""
    membership = list(range(len(adjacency)))
    for _ in range(12):
        vertices = list(range(len(adjacency)))
        degrees = [sum(row.values()) for row in adjacency]
        total = sum(degrees)
        if not total:
            break

        partition = vertices.copy()
        volume = degrees.copy()
        for _ in range(40):
            rng.shuffle(vertices)
            moved = False
            for vertex in vertices:
                previous = partition[vertex]
                volume[previous] -= degrees[vertex]
                neighboring = Counter()
                for other, weight in adjacency[vertex].items():
                    if other != vertex:
                        neighboring[partition[other]] += weight
                penalty = resolution * degrees[vertex] / total
                best = previous
                best_score = neighboring[previous] - volume[previous] * penalty
                for group, weight in sorted(neighboring.items()):
                    score = weight - volume[group] * penalty
                    if score > best_score + 1e-9:
                        best, best_score = group, score
                partition[vertex] = best
                volume[best] += degrees[vertex]
                moved |= best != previous
            if not moved:
                break

        group_ids = {}
        contraction = [group_ids.setdefault(group, len(group_ids)) for group in partition]
        membership = [contraction[group] for group in membership]
        if len(group_ids) == len(adjacency):
            break
        aggregated = [Counter() for _ in group_ids]
        for vertex, row in enumerate(adjacency):
            for other, weight in row.items():
                aggregated[contraction[vertex]][contraction[other]] += weight
        adjacency = aggregated
    return membership


def force_positions(executable, widths, edges, seed):
    """Pass numeric indices and weights to Graphviz, then read its plain coordinates."""
    if len(widths) <= 1:
        return [(0.0, 0.0)] * len(widths)
    dot = [
        "strict graph atlas {",
        f"graph [start={seed}, overlap=prism, maxiter=350];",
        'node [shape=circle, fixedsize=true, label=""];',
    ]
    dot.extend(f"n{i} [width={width:.5f}];" for i, width in enumerate(widths))
    dot.extend(f"n{a} -- n{b} [weight={weight:.5f}];" for a, b, weight in edges)
    dot.append("}")
    try:
        result = subprocess.run(
            [executable, "-Tplain"],
            input="\n".join(dot),
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Graphviz layout timed out. Try a smaller overview graph.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Graphviz could not lay out the graph") from exc
    xy = {}
    for line in result.stdout.splitlines():
        if line.startswith("node "):
            _, node, x, y, *_ = line.split()
            xy[int(node[1:])] = (float(x), float(y))
    if len(xy) != len(widths):
        raise ValueError("Graphviz returned an incomplete layout")
    return [xy[i] for i in range(len(widths))]


def normalize(points, target_radius):
    """Center a local shape and fit it in a circle of the specified radius."""
    x, y = zip(*points)
    center_x = (min(x) + max(x)) / 2
    center_y = (min(y) + max(y)) / 2
    centered = [(px - center_x, py - center_y) for px, py in points]
    radius = max(math.hypot(px, py) for px, py in centered)
    scale = target_radius / radius if radius else 1.0
    return [(px * scale, py * scale) for px, py in centered]


def display_groups(adjacency, resolution, rng):
    partition = modularity_communities(adjacency, resolution, rng)
    members = defaultdict(list)
    for vertex, group in enumerate(partition):
        # Shared color for isolate records. Each isolate remains its own graph vertex.
        members[group if adjacency[vertex] else -1].append(vertex)
    return sorted(members.items(), key=lambda part: (-len(part[1]), part[1][0]))


def place_groups(adjacency, groups, executable, seed):
    """Lay out the group-to-group graph, then lay out each group's internal graph."""
    group_of = {
        vertex: group for group, (_, members) in enumerate(groups) for vertex in members
    }
    bridges = Counter()
    for vertex, row in enumerate(adjacency):
        for other, weight in row.items():
            left, right = group_of[vertex], group_of[other]
            if left < right:
                bridges[left, right] += weight

    radii = [max(0.18, 0.13 * math.sqrt(len(members))) for _, members in groups]
    centers = force_positions(
        executable,
        [radius * 2.35 for radius in radii],
        ((left, right, math.sqrt(ties)) for (left, right), ties in sorted(bridges.items())),
        seed,
    )
    positions = [(0.0, 0.0)] * len(adjacency)
    for group, (_, members) in enumerate(groups):
        local = {vertex: i for i, vertex in enumerate(members)}
        internal_edges = (
            (local[vertex], local[other], math.sqrt(weight))
            for vertex in members
            for other, weight in adjacency[vertex].items()
            if other in local and vertex < other
        )
        shape = force_positions(executable, [0.03] * len(members), internal_edges, seed)
        center_x, center_y = centers[group]
        for vertex, (x, y) in zip(members, normalize(shape, radii[group])):
            # Graphviz measures Y upwards. The browser measures it downwards.
            positions[vertex] = (x + center_x, -y - center_y)
    return normalize(positions, 500) if positions else positions


def layout_graph(graph, *, resolution=1.0, seed=19):
    """Copy the graph, replacing display groups, coordinates and displayed degrees."""
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Community resolution must be a positive finite number")
    if type(seed) is not int or seed <= 0 or seed > 2**31 - 1:
        raise ValueError("Seed must be a positive 32-bit integer")

    meta = graph.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Graph metadata must be an object")

    adjacency, degree = index_adjacency(graph)
    groups = display_groups(adjacency, resolution, random.Random(seed))
    executable = shutil.which("sfdp")
    if not executable and len(adjacency) > 1:
        raise ValueError("Install Graphviz and ensure sfdp is on PATH")
    points = place_groups(adjacency, groups, executable, seed)
    layout_nodes = [node.copy() for node in graph["nodes"]]
    layout_groups = []
    for group, (origin, members) in enumerate(groups):
        representative = min(members, key=lambda vertex: (-degree[vertex], vertex))
        record = layout_nodes[representative]
        layout_groups.append({
            "id": group,
            "label": "Isolated records" if origin == -1 else record.get("label") or record["id"],
            "color": PLACEHOLDER if origin == -1 else PALETTE[group % len(PALETTE)],
        })
        for vertex in members:
            x, y = points[vertex]
            layout_nodes[vertex].update(g=group, x=round(x, 3), y=round(y, 3), d=degree[vertex])

    meta = {
        **meta,
        "layout": "Weighted modularity communities with a multilevel force-directed layout",
        "layout_seed": seed,
    }
    return {**graph, "nodes": layout_nodes, "groups": layout_groups, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=float, default=1.0, help="Community detection resolution")
    parser.add_argument("--seed", type=int, default=19, help="Fixed layout seed")
    args = parser.parse_args()
    try:
        graph = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(graph, dict):
            raise ValueError("Graph JSON must be an object")
        result = layout_graph(graph, resolution=args.resolution, seed=args.seed)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as folder:
            staged = Path(folder) / "layout.json"
            with staged.open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
            staged.replace(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
