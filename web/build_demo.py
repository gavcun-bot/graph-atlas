"""Generate the Graph Atlas demo with deterministic, fictional research records."""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Community:
    key: str
    label: str
    color: str
    station: str
    center: tuple[int, int]
    field: str
    campaigns: int


COMMUNITIES = (
    Community("ocean", "Ocean signals", "#196faa", "Ocean Signal Observatory", (-255, -64), "surface current", 12),
    Community("urban", "Urban ecology", "#9c4374", "Urban Habitat Observatory", (227, -235), "canopy temperature", 8),
    Community("forest", "Forest canopy", "#419053", "Canopy Light Observatory", (59, 151), "understory light", 13),
    Community("water", "Water quality", "#009796", "Headwaters Sampling Station", (-352, 125), "dissolved oxygen", 9),
    Community("ice", "Glacial change", "#6b50b0", "Glacier Margin Observatory", (425, -59), "ice displacement", 10),
    Community("coast", "Coastal sediment", "#cd7133", "Barrier Coast Observatory", (306, 131), "sediment transport", 10),
    Community("soil", "Soil carbon", "#bd5766", "Prairie Soil Observatory", (-128, 237), "soil respiration", 8),
    Community("air", "Atmosphere", "#5685bd", "Atmospheric Sounding Station", (-72, -194), "boundary layer", 9),
)

BRIDGES = (
    ("ocean", "coast", "shoreline exchange"),
    ("ocean", "air", "wind-driven surface transport"),
    ("ocean", "water", "river and coastal exchanges"),
    ("water", "ice", "seasonal meltwater"),
    ("water", "forest", "catchment conditions"),
    ("coast", "ice", "sediment deposition"),
    ("urban", "air", "surface warming"),
    ("urban", "forest", "canopy conditions"),
    ("forest", "soil", "vegetation and soil activity"),
    ("soil", "air", "near-surface carbon exchange"),
    ("air", "ice", "high elevation conditions"),
    ("coast", "forest", "coastal canopy exposure"),
)


def build_graph(seed=21):
    """Return a JSON-ready graph with array-index edge endpoints."""
    rng = random.Random(seed)
    nodes, edges, groups = [], [], []
    stations, protocol_indices, dataset_indices = {}, {}, {}

    def record(kind, key, label, description, group_id, center, spread=0):
        index = len(nodes)
        if spread:
            x = center[0] + rng.gauss(0, spread)
            y = center[1] + rng.gauss(0, spread * 0.82)
        else:
            x, y = center
        nodes.append({
            "id": f"{kind.lower()}:{key}",
            "label": label,
            "type": kind,
            "detail": description,
            "source": "Fictional field research catalogue",
            "g": group_id,
            "d": 0,
            "x": x,
            "y": y,
        })
        return index

    def relate(source, target, relationship, context):
        edges.append([source, target, relationship, context])
        nodes[source]["d"] += 1
        nodes[target]["d"] += 1

    for gid, group in enumerate(COMMUNITIES):
        groups.append({"id": gid, "label": group.label, "color": group.color})
        key = group.key
        stations[key] = record(
            "Station", f"{key}-01", group.station,
            f"Fictional station coordinating the {group.field} research programme.",
            gid, group.center,
        )
        base_angle = rng.random() * math.tau
        protocols, datasets, sensors = [], [], []
        protocol_indices[key] = protocols
        dataset_indices[key] = datasets

        for campaign in range(1, group.campaigns + 1):
            # Protocols occupy small neighborhoods around the station. A fixed
            # seed keeps the layout stable across re-exports of the same graph.
            angle = base_angle + (campaign - 1) * math.tau / group.campaigns
            distance = rng.uniform(46, 96)
            branch_center = (
                group.center[0] + distance * math.cos(angle),
                group.center[1] + distance * math.sin(angle),
            )
            suffix = f"{campaign:02d}"
            branch_key = f"{key}-{suffix}"
            title = f"{group.label} · {{}} {suffix}"
            protocol = record(
                "Protocol", branch_key, title.format("Sampling protocol"),
                f"Fictional protocol specifying the sampling checklist for {group.field}.",
                gid, branch_center, 5,
            )
            dataset = record(
                "Dataset", branch_key, title.format("Observation series"),
                f"Generated records representing one campaign's {group.field} measurements.",
                gid, branch_center, 12,
            )
            sensor = record(
                "Sensor", branch_key, title.format("Field sensor"),
                f"Fictional sensor configuration used to illustrate {group.field} measurement.",
                gid, branch_center, 14,
            )
            report = record(
                "Report", branch_key, title.format("Campaign report"),
                "Fictional field report with a protocol reference and links to observations.",
                gid, branch_center, 22,
            )
            protocols.append(protocol)
            datasets.append(dataset)
            sensors.append(sensor)
            relate(stations[key], protocol, "uses protocol", f"Station programme, campaign {suffix}.")
            relate(protocol, dataset, "defines sampling for", f"Observation series {suffix} cites its protocol.")
            relate(protocol, sensor, "specifies sensor", f"Equipment checklist for campaign {suffix}.")
            relate(report, dataset, "summarizes dataset", f"Fictional field report {suffix}, results section.")
            relate(report, protocol, "describes method", f"Fictional field report {suffix}, methods section.")
            relate(stations[key], sensor, "operates sensor", f"Programme sensor roster, entry {suffix}.")

            for seq in range(1, rng.randint(8, 12)):
                obs_key = f"{branch_key}-{seq:02d}"
                observation = record(
                    "Observation", obs_key, f"{group.label} · Observation {suffix}.{seq:02d}",
                    f"Generated {group.field} observation from campaign {suffix}.",
                    gid, branch_center, rng.uniform(13, 32),
                )
                relate(dataset, observation, "contains observation", f"Series {suffix}, entry {seq:02d}.")
                if seq % 3:
                    relate(observation, sensor, "recorded with", f"Observation {seq:02d}, instrument field.")
                if seq % 4 == 0:
                    relate(report, observation, "highlights observation", f"Fictional report {suffix}, example entry {seq:02d}.")

            for seq in range(1, 3):
                sample = record(
                    "Sample", f"{branch_key}-{seq:02d}", f"{group.label} · Sample {suffix}.{seq:02d}",
                    f"Generated field sample associated with the {group.field} campaign.",
                    gid, branch_center, 19,
                )
                relate(protocol, sample, "governs collection of", f"Sampling log for campaign {suffix}, record {seq:02d}.")
                relate(dataset, sample, "records sample", f"Series {suffix}, sample inventory.")

        for index, protocol in enumerate(protocols):
            neighbor = (index + 1) % len(protocols)
            relate(protocol, datasets[neighbor], "shares checklist with", "The fictional catalogue groups adjacent campaigns by protocol revision.")
            if index % 2 == 0:
                relate(datasets[index], sensors[neighbor], "compares measurements from", "Illustrative consistency check across field sensors.")

    for key, neighbor, subject in BRIDGES:
        relate(stations[key], stations[neighbor], "collaborates with", f"Fictional research programme on {subject}.")
        for j in range(4):
            left = dataset_indices[key][j]
            right_series = dataset_indices[neighbor]
            right = right_series[(j * 2 + 1) % len(right_series)]
            protocol = protocol_indices[neighbor][j]
            relate(left, right, "compared against", f"Generated joint comparison on {subject}.")
            relate(left, protocol, "references sampling method", f"Method cross-reference in the fictional {subject} study.")

    relax_positions(nodes, edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "meta": {
            "scope": "Fictional field research data. Search Ocean Signal Observatory, open a connection, and follow its evidence to a related record.",
        },
    }


def relax_positions(nodes, edges):
    """Pull connected records toward one another from stable group positions."""
    initial = [(node["x"], node["y"]) for node in nodes]
    for _ in range(16):
        offsets = [[0.0, 0.0] for _ in nodes]
        for a, b, _, _ in edges:
            first, second = nodes[a], nodes[b]
            delta_x, delta_y = second["x"] - first["x"], second["y"] - first["y"]
            length = math.hypot(delta_x, delta_y)
            if length == 0:
                continue
            between_groups = first["g"] != second["g"]
            desired = 150 if between_groups else 18
            weight = 0.035 if between_groups else 0.13
            strength = weight * (length - desired) / length
            for endpoint, direction in ((a, 1), (b, -1)):
                offsets[endpoint][0] += direction * delta_x * strength
                offsets[endpoint][1] += direction * delta_y * strength

        for node, (dx, dy), (home_x, home_y) in zip(nodes, offsets, initial):
            resistance = 0.8 * math.sqrt(max(1, node["d"]))
            node["x"] += dx / resistance + (home_x - node["x"]) * 0.22
            node["y"] += dy / resistance + (home_y - node["y"]) * 0.22

    for node in nodes:
        node["x"], node["y"] = round(node["x"], 2), round(node["y"], 2)


def build_html(graph, template):
    """Embed graph data so the generated demo works offline and from file://."""
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", r"\u003c").replace("\u2028", r"\u2028").replace("\u2029", r"\u2029")
    if template.count("__GRAPH_DATA__") != 1:
        raise ValueError("Expected one embedded graph placeholder in the viewer template.")
    return template.replace("__GRAPH_DATA__", payload)


def main():
    web_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional positioned graph JSON to embed")
    parser.add_argument("--output", type=Path, default=web_dir.parent / "docs" / "index.html", help="Output HTML file")
    parser.add_argument("--json", type=Path, help="Optional graph JSON export")
    parser.add_argument("--seed", type=int, default=21, help="Seed used for synthetic sampling")
    args = parser.parse_args()
    graph = build_graph(args.seed) if args.input is None else json.loads(args.input.read_text(encoding="utf-8"))
    template = (web_dir / "template.html").read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(graph, template), encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(graph['nodes']):,} nodes and {len(graph['edges']):,} links: {args.output}")


if __name__ == "__main__":
    main()
