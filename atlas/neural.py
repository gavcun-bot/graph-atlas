"""Learn and query structural similarity from a Graph Atlas SQLite database."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    import numpy as np
except ImportError:
    np = None


def read_graph(path):
    """Build a sorted, undirected adjacency index from recorded endpoints."""
    with closing(sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as db:
        records = {
            identifier: (label, kind)
            for identifier, label, kind in db.execute("SELECT id, label, type FROM nodes")
        }
        pairs = {
            tuple(sorted((source, target)))
            for source, target in db.execute("SELECT source, target FROM edges WHERE source != target")
            if source in records and target in records
        }

    ids = sorted({identifier for pair in pairs for identifier in pair})
    if len(ids) < 3:
        raise ValueError("The graph needs at least three nodes with links")

    index = {identifier: number for number, identifier in enumerate(ids)}
    left, right = np.array([(index[a], index[b]) for a, b in pairs], dtype=np.int64).T
    sources = np.concatenate((left, right))
    targets = np.concatenate((right, left))
    neighbors = targets[np.lexsort((targets, sources))]
    offsets = np.r_[0, np.cumsum(np.bincount(sources, minlength=len(ids)))]
    return ids, records, neighbors, offsets


def component_labels(neighbors, offsets):
    components = np.full(len(offsets) - 1, -1, dtype=np.int64)
    component = 0
    for index in range(len(components)):
        if components[index] >= 0:
            continue
        components[index] = component
        stack = [index]
        while stack:
            vertex = stack.pop()
            for neighbor in neighbors[offsets[vertex]:offsets[vertex + 1]]:
                if components[neighbor] < 0:
                    components[neighbor] = component
                    stack.append(neighbor)
        component += 1
    return components


def sample_context(rng, neighbors, offsets, centers, negatives):
    def walk_step(vertices):
        start = offsets[vertices]
        degree = offsets[vertices + 1] - start
        return neighbors[start + (rng.random(len(vertices)) * degree).astype(np.int64)]

    direct = walk_step(centers)
    two_hop = walk_step(direct)
    positives = np.where((rng.random(len(centers)) < 0.5) & (two_hop != centers), two_hop, direct)

    # Draw uniformly from all entities except the center and its sampled context.
    # Shift samples past those two excluded indices instead of re-sampling collisions.
    lower = np.minimum(centers, positives)[:, None]
    upper = np.maximum(centers, positives)[:, None]
    noise = rng.integers(0, len(offsets) - 3, size=(len(centers), negatives))
    noise += noise >= lower
    noise += noise >= upper
    return np.column_stack((positives, noise))


def logistic_loss(w_in, w_out, centers, targets):
    logits = np.einsum("ij,ikj->ik", w_in[centers], w_out[targets])
    correct_sign = logits.copy()
    correct_sign[:, 0] *= -1
    return float(np.logaddexp(0, correct_sign).mean())


def write_model(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        staged_path = Path(temporary.name)
    try:
        np.savez_compressed(staged_path, **fields)
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def train(args):
    if args.output.suffix.lower() != ".npz" or args.output.resolve() == args.database.resolve():
        raise ValueError("Choose a separate .npz model output")
    ids, records, neighbors, offsets = read_graph(args.database)

    rng = np.random.default_rng(args.seed)
    size, dimensions = len(ids), args.dimensions
    w_in = rng.normal(0, 0.06, size=(size, dimensions)).astype(np.float32)
    w_out = rng.normal(0, 0.06, size=(size, dimensions)).astype(np.float32)
    diagnostic = rng.integers(0, size, size=min(8192, 8 * size))
    diagnostic_context = sample_context(rng, neighbors, offsets, diagnostic, args.negatives)
    before = logistic_loss(w_in, w_out, diagnostic, diagnostic_context)

    visits = 8
    visits_index = np.tile(np.arange(size), visits)
    batch_size = 192
    for epoch in range(args.epochs):
        centers = rng.permutation(visits_index)
        targets = sample_context(rng, neighbors, offsets, centers, args.negatives)
        rate = args.rate * (1 - 0.65 * epoch / max(1, args.epochs - 1))
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]
            input_vectors = w_in[batch_centers]
            context_vectors = w_out[batch_targets]
            logits = np.einsum("ij,ikj->ik", input_vectors, context_vectors)
            error = -np.exp(-np.logaddexp(0, -logits))
            error[:, 0] += 1
            input_gradient = np.einsum("ik,ikj->ij", error, context_vectors)
            context_gradient = error[:, :, None] * input_vectors[:, None, :]
            np.add.at(w_in, batch_centers, rate * input_gradient)
            np.add.at(w_out, batch_targets.ravel(), rate * context_gradient.reshape(-1, dimensions))

    after = logistic_loss(w_in, w_out, diagnostic, diagnostic_context)
    if not (np.isfinite(after) and np.isfinite(w_in).all() and np.isfinite(w_out).all()):
        raise ValueError("Training produced non-finite weights. Try a lower learning rate.")

    components = component_labels(neighbors, offsets)
    report = {
        "model": "Skip-gram with one- and two-hop contexts and negative sampling",
        "nodes": size,
        "unique_undirected_pairs": len(neighbors) // 2,
        "connected_components": int(components.max()) + 1,
        "dimensions": dimensions,
        "epochs": args.epochs,
        "seed": args.seed,
        "numpy": np.__version__,
        "negatives": args.negatives,
        "initial_learning_rate": args.rate,
        "visits_per_node_per_epoch": visits,
        "diagnostic_loss_before": before,
        "diagnostic_loss_after": after,
        "diagnostic_scope": "Fixed sample drawn from the training graph",
        "interpretation": "Cosine similarity reflects graph structure and suggests entities to investigate.",
    }
    write_model(
        args.output,
        ids=np.array(ids),
        labels=np.array([records[identifier][0] for identifier in ids]),
        types=np.array([records[identifier][1] for identifier in ids]),
        components=components,
        embedding_in=w_in,
        embedding_out=w_out,
        metadata=np.array(json.dumps(report, sort_keys=True)),
    )
    return report


def related(model, identifier, limit, same_type):
    ids = model["ids"]
    matches = np.flatnonzero(ids == identifier)
    if len(matches) == 0:
        raise ValueError(f"No embedding for {identifier!r}. Check the entity ID or training graph.")

    index = matches[0]
    vectors = model["embedding_in"] + model["embedding_out"]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    components = model["components"]
    kinds = model["types"]
    allowed = (components == components[index]) & (ids != identifier)
    if same_type:
        allowed &= kinds == kinds[index]

    candidates = np.flatnonzero(allowed)
    scores = vectors[candidates] @ vectors[index]
    order = np.argsort(-scores, kind="stable")[:limit]
    labels = model["labels"]
    return [
        {
            "id": str(ids[candidates[position]]),
            "label": str(labels[candidates[position]]),
            "type": str(kinds[candidates[position]]),
            "cosine_similarity": round(float(scores[position]), 4),
        }
        for position in order
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train from a Graph Atlas SQLite database")
    training.add_argument("database", type=Path)
    training.add_argument("--output", type=Path, required=True, help="Path for the model .npz")
    training.add_argument("--dimensions", type=int, default=32)
    training.add_argument("--seed", type=int, default=73017)
    training.add_argument("--epochs", type=int, default=32)
    training.add_argument("--negatives", type=int, default=4)
    training.add_argument("--rate", type=float, default=0.058)

    query = commands.add_parser("related", help="Rank entities by cosine similarity within a component")
    query.add_argument("model", type=Path)
    query.add_argument("id")
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--same-type", action="store_true")
    args = parser.parse_args()

    if np is None:
        parser.exit(2, "Neural training and similarity require NumPy. Install it with: python -m pip install numpy\n")

    try:
        if args.command == "train":
            if min(args.epochs, args.dimensions, args.negatives) < 1 or args.seed < 0:
                parser.error("Epochs, dimensions and negatives must be positive. The seed must be non-negative.")
            if not np.isfinite(args.rate) or args.rate <= 0:
                parser.error("Learning rate must be finite and positive")
            result = train(args)
        else:
            if args.limit < 1:
                parser.error("Limit must be positive")
            with np.load(args.model, allow_pickle=False) as model:
                result = related(model, args.id, args.limit, args.same_type)
    except (OSError, KeyError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"{error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
