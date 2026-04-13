#!/usr/bin/env python3
"""
Download and prepare NELL-995 for SGKU/SDKU experiments.

Output dataset layout (default):
  data/NELL-995-20/
    triples.txt
    timesteps/{0,1,2}.txt
    schema.txt
    schema_store.json
    entity_types.json
    entity_mappings.json
    relation_mappings.json
    relation_to_pattern.json
    relation_names.txt
    metadata.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


RAW_BASE = "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/NELL-995"
REQUIRED_FILES = [
    "entity2id.txt",
    "relation2id.txt",
    "train2id.txt",
    "valid2id.txt",
    "test2id.txt",
    "type_constrain.txt",
]


def _parse_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        _ = int(first) if first else 0
        name_to_id: dict[str, int] = {}
        id_to_name: dict[int, str] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    continue
            name, idx_str = parts[0], parts[1]
            idx = int(idx_str)
            name_to_id[name] = idx
            id_to_name[idx] = name
    return name_to_id, id_to_name


def _parse_triple_ids(path: Path) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        _ = int(first) if first else 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            h_str, t_str, r_str = line.split()
            triples.append((int(h_str), int(r_str), int(t_str)))
    return triples


def _parse_type_constraints(path: Path) -> dict[int, dict[str, list[int]]]:
    """
    OpenKE type_constrain format:
      first line: relation_count
      then two lines per relation id:
        rel_id  n  ent_id_1 ent_id_2 ...
        rel_id  m  ent_id_1 ent_id_2 ...
    We interpret first as allowed heads and second as allowed tails.
    """
    out: dict[int, dict[str, list[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        _ = int(first) if first else 0
        by_rel: dict[int, list[list[int]]] = defaultdict(list)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_id = int(parts[0])
            count = int(parts[1])
            ids = [int(x) for x in parts[2 : 2 + count]]
            by_rel[rel_id].append(ids)
    for rid, lst in by_rel.items():
        heads = lst[0] if len(lst) >= 1 else []
        tails = lst[1] if len(lst) >= 2 else []
        out[rid] = {"head_ids": heads, "tail_ids": tails}
    return out


def _infer_entity_type(entity_name: str) -> str:
    # NELL entities are commonly formatted like: concept_person_xxx or concept:person:xxx
    if entity_name.startswith("concept_"):
        parts = entity_name.split("_", 2)
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if entity_name.startswith("concept:"):
        parts = entity_name.split(":", 2)
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if "_" in entity_name:
        return entity_name.split("_", 1)[0]
    return "Entity"


def _prepare_schema_and_types(
    triples: list[tuple[int, int, int]],
    id2entity: dict[int, str],
    id2relation: dict[int, str],
    type_constraints: dict[int, dict[str, list[int]]] | None = None,
    min_type_match: float = 0.2,
) -> tuple[dict[int, str], dict[str, tuple[str, str]], dict[tuple[str, str, str], dict]]:
    entity_types: dict[int, str] = {}
    for eid, name in id2entity.items():
        entity_types[eid] = _infer_entity_type(name)

    # Infer relation domain/range by majority type from observed triples.
    rel_head_type_counter: dict[str, Counter] = defaultdict(Counter)
    rel_tail_type_counter: dict[str, Counter] = defaultdict(Counter)
    rel_count: Counter = Counter()
    for h, r, t in triples:
        rel_name = id2relation[r]
        rel_count[rel_name] += 1
        rel_head_type_counter[rel_name][entity_types[h]] += 1
        rel_tail_type_counter[rel_name][entity_types[t]] += 1

    # Optional relation type priors from OpenKE type constraints.
    rel_head_constraint_counter: dict[str, Counter] = defaultdict(Counter)
    rel_tail_constraint_counter: dict[str, Counter] = defaultdict(Counter)
    if type_constraints:
        for rid, sides in type_constraints.items():
            rel_name = id2relation.get(rid)
            if rel_name is None:
                continue
            for eid in sides.get("head_ids", []):
                if eid in entity_types:
                    rel_head_constraint_counter[rel_name][entity_types[eid]] += 1
            for eid in sides.get("tail_ids", []):
                if eid in entity_types:
                    rel_tail_constraint_counter[rel_name][entity_types[eid]] += 1

    relation_to_types: dict[str, tuple[str, str]] = {}
    schema_store: dict[tuple[str, str, str], dict] = {}

    # Build schema_store with entity ID sets for SGKU direct loader.
    pattern_heads: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    pattern_tails: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    pattern_rels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    pattern_support: Counter = Counter()

    for h, r, t in triples:
        rel_name = id2relation[r]
        dom = (
            rel_head_constraint_counter[rel_name].most_common(1)[0][0]
            if rel_head_constraint_counter[rel_name]
            else rel_head_type_counter[rel_name].most_common(1)[0][0]
        )
        rng = (
            rel_tail_constraint_counter[rel_name].most_common(1)[0][0]
            if rel_tail_constraint_counter[rel_name]
            else rel_tail_type_counter[rel_name].most_common(1)[0][0]
        )
        # Coherence guard: domain/range must appear in observed triples for this relation.
        if rel_head_type_counter[rel_name].get(dom, 0) <= 0:
            dom = rel_head_type_counter[rel_name].most_common(1)[0][0]
        if rel_tail_type_counter[rel_name].get(rng, 0) <= 0:
            rng = rel_tail_type_counter[rel_name].most_common(1)[0][0]
        # Low-confidence guard: if selected type matches too few triples, use Entity.
        n = max(1, int(rel_count[rel_name]))
        dom_match = rel_head_type_counter[rel_name].get(dom, 0) / n
        rng_match = rel_tail_type_counter[rel_name].get(rng, 0) / n
        if dom_match < float(min_type_match):
            dom = "Entity"
        if rng_match < float(min_type_match):
            rng = "Entity"

        relation_to_types[rel_name] = (dom, rng)
        key = (dom, rel_name, rng)
        pattern_heads[key].add(h)
        pattern_tails[key].add(t)
        pattern_rels[key].add(rel_name)
        pattern_support[key] += 1

    max_support = max(pattern_support.values()) if pattern_support else 1
    for key, support in pattern_support.items():
        # 0.5..2.0 importance range using smoothed log support.
        importance = 0.5 + 1.5 * (math.log1p(support) / math.log1p(max_support))
        schema_store[key] = {
            "importance": round(float(importance), 4),
            "entities": {
                "head": sorted(pattern_heads[key]),
                "tail": sorted(pattern_tails[key]),
            },
            "relations": sorted(pattern_rels[key]),
        }

    return entity_types, relation_to_types, schema_store


def _write_timesteps(
    out_dir: Path,
    triples_lines: list[str],
    *,
    steps: int,
    percentage: int,
    seed: int,
) -> None:
    timesteps_dir = out_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    remaining = list(triples_lines)
    rng.shuffle(remaining)
    remove_each = int(len(remaining) * (percentage / 100.0))
    if remove_each <= 0:
        remove_each = max(1, len(remaining) // max(steps, 1))

    for step in range(steps):
        if not remaining:
            break
        cut = min(remove_each, len(remaining))
        remaining = remaining[cut:]
        with (timesteps_dir / f"{step}.txt").open("w", encoding="utf-8") as f:
            for line in remaining:
                f.write(line + "\n")


def _download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)


def prepare_nell995_dataset(
    out_dir: Path, steps: int, percentage: int, seed: int, min_type_match: float = 0.2
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw_openke"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname in REQUIRED_FILES:
        url = f"{RAW_BASE}/{fname}"
        target = raw_dir / fname
        print(f"Downloading {url}")
        _download_file(url, target)

    entity2id, id2entity = _parse_mapping(raw_dir / "entity2id.txt")
    raw_relation2id, raw_id2relation = _parse_mapping(raw_dir / "relation2id.txt")
    type_constraints = _parse_type_constraints(raw_dir / "type_constrain.txt")
    train = _parse_triple_ids(raw_dir / "train2id.txt")
    valid = _parse_triple_ids(raw_dir / "valid2id.txt")
    test = _parse_triple_ids(raw_dir / "test2id.txt")
    all_triples = train + valid + test

    # IMPORTANT: forward relations must use even IDs so inverse relations are +1 (odd IDs).
    relation_names = sorted(raw_relation2id.keys())
    relation2id = {rel: 2 * i for i, rel in enumerate(relation_names)}
    id2relation = {idx: rel for rel, idx in relation2id.items()}

    triples_lines: list[str] = []
    for h, r, t in all_triples:
        rel_name = raw_id2relation[r]
        triples_lines.append(f"{id2entity[h]}\t{rel_name}\t{id2entity[t]}")

    # Write merged triples for current codebase format.
    (out_dir / "triples.txt").write_text("\n".join(triples_lines) + "\n", encoding="utf-8")

    entity_types, relation_to_types, schema_store = _prepare_schema_and_types(
        all_triples,
        id2entity,
        raw_id2relation,
        type_constraints=type_constraints,
        min_type_match=min_type_match,
    )

    # schema.txt
    with (out_dir / "schema.txt").open("w", encoding="utf-8") as f:
        for rel_name, (dom, rng) in sorted(relation_to_types.items()):
            f.write(f"{dom} {rel_name} {rng}\n")

    # entity_types.json: entity-name -> [type]
    entity_types_json = {id2entity[eid]: [etype] for eid, etype in entity_types.items()}
    (out_dir / "entity_types.json").write_text(
        json.dumps(entity_types_json, indent=2), encoding="utf-8"
    )

    # entity/relation mappings
    entity_mappings = {
        "entity_to_id": entity2id,
        "id_to_entity": {str(k): v for k, v in id2entity.items()},
    }
    (out_dir / "entity_mappings.json").write_text(
        json.dumps(entity_mappings, indent=2), encoding="utf-8"
    )

    relation_schema = {
        rel: {"domain": dom, "range": rng}
        for rel, (dom, rng) in relation_to_types.items()
    }
    relation_mappings = {
        "relation_to_id": relation2id,
        "id_to_relation": {str(k): v for k, v in id2relation.items()},
        "relation_to_types": relation_to_types,
        "relation_schema": relation_schema,
    }
    (out_dir / "relation_mappings.json").write_text(
        json.dumps(relation_mappings, indent=2), encoding="utf-8"
    )

    (out_dir / "relation_names.txt").write_text(
        "\n".join(sorted(relation2id.keys())) + "\n", encoding="utf-8"
    )

    relation_to_pattern = {}
    for (dom, rel, rng), _payload in schema_store.items():
        relation_to_pattern[rel] = [dom, rel, rng]
    (out_dir / "relation_to_pattern.json").write_text(
        json.dumps(relation_to_pattern, indent=2), encoding="utf-8"
    )

    schema_store_json = {}
    for key, payload in schema_store.items():
        schema_store_json[str(key)] = payload
    (out_dir / "schema_store.json").write_text(
        json.dumps(schema_store_json, indent=2), encoding="utf-8"
    )

    _write_timesteps(out_dir, triples_lines, steps=steps, percentage=percentage, seed=seed)

    metadata_yaml = (
        f"dataset: {out_dir.name}\n"
        f"source: OpenKE NELL-995\n"
        f"raw_base_url: {RAW_BASE}\n"
        f"triples_merged: {len(all_triples)}\n"
        f"entities: {len(entity2id)}\n"
        f"relations: {len(relation2id)}\n"
        f"timesteps_num: {steps}\n"
        f"per_step_percentage: {percentage}\n"
        f"seed: {seed}\n"
    )
    (out_dir / "metadata.yaml").write_text(metadata_yaml, encoding="utf-8")

    print(f"Prepared dataset at: {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare NELL-995 dataset.")
    parser.add_argument(
        "--dataset",
        default="NELL-995-20",
        help="Output dataset folder name under data/ (default: NELL-995-20).",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of timesteps.")
    parser.add_argument(
        "--percentage",
        type=int,
        default=20,
        help="Percentage removed at each timestep to build timesteps/{i}.txt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-type-match",
        type=float,
        default=0.2,
        help="Minimum per-relation type match ratio; otherwise domain/range becomes Entity.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / args.dataset
    prepare_nell995_dataset(
        output_dir,
        steps=args.steps,
        percentage=args.percentage,
        seed=args.seed,
        min_type_match=float(args.min_type_match),
    )
