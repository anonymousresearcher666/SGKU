#!/usr/bin/env python3
"""
Download and prepare CoDEx datasets for this codebase.

Creates folders such as:
  data/CoDEx-S-20
  data/CoDEx-M-20
  data/CoDEx-L-20

Each folder includes:
  - triples.txt
  - timesteps/{0,1,2}.txt
  - schema.txt
  - schema_store.json
  - entity_types.json
  - entity_mappings.json
  - relation_mappings.json
  - relation_to_pattern.json
  - relation_names.txt
  - metadata.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


BASE_RAW = "https://raw.githubusercontent.com/tsafavi/codex/master/data"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIDATA_WBGET_URL = "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=labels|descriptions&languages=en&ids={ids}"
WIKIDATA_USER_AGENT = "KGUNLEARNING/1.0 (research schema builder)"

# Wikidata property-constraint types
CONSTRAINT_SUBJECT_TYPE = "Q21503250"  # subject type constraint
CONSTRAINT_VALUE_TYPE = "Q21510865"  # value type constraint

_ENTITY_CACHE: dict[str, dict] = {}
_LABEL_CACHE: dict[str, str] = {}


def _download_text(url: str, retries: int = 3) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": WIKIDATA_USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - network failure path
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Failed download for URL: {url}") from last_exc


def _download_json(url: str, retries: int = 3) -> dict:
    return json.loads(_download_text(url, retries=retries))


def _load_codex_split(split_url: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    raw = _download_text(split_url)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            parts = line.split()
            if len(parts) != 3:
                continue
        out.append((parts[0], parts[1], parts[2]))
    return out


def _write_timesteps(out_dir: Path, triples_lines: list[str], *, steps: int, percentage: int, seed: int) -> None:
    ts_dir = out_dir / "timesteps"
    ts_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    remaining = list(triples_lines)
    rng.shuffle(remaining)
    remove_each = int(len(remaining) * (percentage / 100.0))
    if remove_each <= 0:
        remove_each = max(1, len(remaining) // max(steps, 1))

    for i in range(steps):
        if not remaining:
            break
        cut = min(remove_each, len(remaining))
        remaining = remaining[cut:]
        with (ts_dir / f"{i}.txt").open("w", encoding="utf-8") as f:
            for line in remaining:
                f.write(line + "\n")


def _primary_type(type_ids: list[str] | None) -> str:
    if not type_ids:
        return "Entity"
    # Keep first Wikidata type id as primary type label.
    return type_ids[0]


def _extract_item_id(value: object) -> str | None:
    if isinstance(value, dict):
        maybe_id = value.get("id")
        if isinstance(maybe_id, str):
            return maybe_id
    return None


def _fetch_wikidata_entity(qid: str) -> dict:
    if qid in _ENTITY_CACHE:
        return _ENTITY_CACHE[qid]
    payload = _download_json(WIKIDATA_ENTITY_URL.format(qid=qid))
    entity = payload.get("entities", {}).get(qid, {})
    _ENTITY_CACHE[qid] = entity
    return entity


def _fetch_labels(qids: list[str]) -> dict[str, str]:
    missing = [q for q in qids if q and q not in _LABEL_CACHE]
    for i in range(0, len(missing), 50):
        chunk = missing[i : i + 50]
        payload = _download_json(WIKIDATA_WBGET_URL.format(ids="|".join(chunk)))
        for qid, entity in payload.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value", qid)
            _LABEL_CACHE[qid] = label
    return {qid: _LABEL_CACHE.get(qid, qid) for qid in qids}


def _get_property_constraints(prop_id: str) -> tuple[list[str], list[str]]:
    """
    Returns:
      (domain_candidates, range_candidates) from Wikidata constraints:
      - subject type constraint (Q21503250) -> domain
      - value type constraint   (Q21510865) -> range
    """
    entity = _fetch_wikidata_entity(prop_id)
    claims = entity.get("claims", {})
    domain_candidates: list[str] = []
    range_candidates: list[str] = []

    for statement in claims.get("P2302", []):
        constraint_id = _extract_item_id(statement.get("mainsnak", {}).get("datavalue", {}).get("value"))
        qualifier_items = [
            _extract_item_id(q.get("datavalue", {}).get("value"))
            for q in statement.get("qualifiers", {}).get("P2308", [])
        ]
        qualifier_items = [qid for qid in qualifier_items if qid]

        if constraint_id == CONSTRAINT_SUBJECT_TYPE:
            domain_candidates.extend(qualifier_items)
        elif constraint_id == CONSTRAINT_VALUE_TYPE:
            range_candidates.extend(qualifier_items)

    # Deduplicate while preserving order.
    domain_candidates = list(dict.fromkeys(domain_candidates))
    range_candidates = list(dict.fromkeys(range_candidates))
    return domain_candidates, range_candidates


def _pick_best_type(candidates: list[str], observed_counts: Counter) -> str | None:
    if not candidates:
        return None
    best_qid = None
    best_count = -1
    for qid in candidates:
        cnt = int(observed_counts.get(qid, 0))
        if cnt > best_count:
            best_count = cnt
            best_qid = qid
    return best_qid if best_count > 0 else None


def _parse_llm_schema_json(raw: str) -> tuple[str | None, str | None]:
    raw = raw.strip()
    if not raw:
        return None, None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        dom = data.get("domain")
        rng = data.get("range")
        qid_re = re.compile(r"^Q[1-9][0-9]{0,9}$")
        dom = dom if isinstance(dom, str) and qid_re.match(dom) else None
        rng = rng if isinstance(rng, str) and qid_re.match(rng) else None
        return dom, rng
    except Exception:
        return None, None


def _llm_fallback_domain_range(
    prop_id: str,
    prop_label: str,
    prop_description: str,
    domain_candidates: list[str],
    range_candidates: list[str],
    observed_head_types: Counter,
    observed_tail_types: Counter,
    llm_model: str,
) -> tuple[str | None, str | None]:
    # Focus the prompt to the top observed classes only (token-safe).
    top_head = [t for t, _ in observed_head_types.most_common(15)]
    top_tail = [t for t, _ in observed_tail_types.most_common(15)]
    candidate_ids = list(dict.fromkeys(domain_candidates + range_candidates + top_head + top_tail))
    labels = _fetch_labels(candidate_ids)

    def _format_types(type_ids: list[str], counts: Counter) -> list[dict[str, object]]:
        out = []
        for tid in type_ids:
            out.append({"id": tid, "label": labels.get(tid, tid), "count": int(counts.get(tid, 0))})
        return out

    llm_input = json.dumps(
        {
            "property_id": prop_id,
            "property_label": prop_label,
            "property_description": prop_description,
            "wikidata_domain_candidates": _format_types(domain_candidates, observed_head_types),
            "wikidata_range_candidates": _format_types(range_candidates, observed_tail_types),
            "observed_head_types": _format_types(top_head, observed_head_types),
            "observed_tail_types": _format_types(top_tail, observed_tail_types),
        }
    )
    system_prompt = (
        "You infer Wikidata domain/range class IDs for a property. "
        "Return ONLY strict JSON: {\"domain\":\"Q...\",\"range\":\"Q...\"}. "
        "Pick IDs that are plausible Wikidata classes."
    )

    allowed_ids = set(candidate_ids)

    def _validate(dom: str | None, rng: str | None) -> tuple[str | None, str | None]:
        dom_ok = dom if dom in allowed_ids else None
        rng_ok = rng if rng in allowed_ids else None
        return dom_ok, rng_ok

    if llm_model.startswith("ollama:"):
        model_name = llm_model.split(":", 1)[1]
        if not shutil.which("ollama"):
            return None, None
        prompt = f"{system_prompt}\n\n{llm_input}"
        try:
            proc = subprocess.run(
                ["ollama", "run", model_name, prompt],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if proc.returncode != 0:
                return None, None
            return _validate(*_parse_llm_schema_json(proc.stdout))
        except Exception:
            return None, None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, None
    payload = {
        "model": llm_model,
        "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": llm_input}],
        "text": {"format": {"type": "text"}},
        "max_output_tokens": 120,
        "temperature": 0.0,
    }
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        res = json.loads(raw)
        return _validate(*_parse_llm_schema_json(res.get("output_text", "")))
    except Exception:
        return None, None


def prepare_codex_dataset(
    size: str,
    out_dir: Path,
    steps: int,
    percentage: int,
    seed: int,
    llm_model: str | None = None,
    min_type_match: float = 0.2,
) -> None:
    size = size.lower()
    if size not in {"s", "m", "l"}:
        raise ValueError("size must be one of: s, m, l")

    triples_prefix = f"{BASE_RAW}/triples/codex-{size}"
    train = _load_codex_split(f"{triples_prefix}/train.txt")
    valid = _load_codex_split(f"{triples_prefix}/valid.txt")
    test = _load_codex_split(f"{triples_prefix}/test.txt")
    triples = train + valid + test

    rel_meta = json.loads(_download_text(f"{BASE_RAW}/relations/en/relations.json"))
    entity2types = json.loads(_download_text(f"{BASE_RAW}/types/entity2types.json"))

    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = {i: e for e, i in entity2id.items()}
    # IMPORTANT: forward relations must use even IDs so inverse relations are +1 (odd IDs).
    relation2id = {r: 2 * i for i, r in enumerate(relations)}
    id2relation = {i: r for r, i in relation2id.items()}

    entity_primary_type = {e: _primary_type(entity2types.get(e, [])) for e in entities}

    # Observed type distributions (all types + primary fallback).
    rel_head_types_all: dict[str, Counter] = defaultdict(Counter)
    rel_tail_types_all: dict[str, Counter] = defaultdict(Counter)
    rel_head_primary: dict[str, Counter] = defaultdict(Counter)
    rel_tail_primary: dict[str, Counter] = defaultdict(Counter)
    rel_count: Counter = Counter()
    for h, r, t in triples:
        rel_count[r] += 1
        rel_head_primary[r][entity_primary_type[h]] += 1
        rel_tail_primary[r][entity_primary_type[t]] += 1
        head_types = entity2types.get(h, []) or [entity_primary_type[h]]
        tail_types = entity2types.get(t, []) or [entity_primary_type[t]]
        for ht in head_types:
            rel_head_types_all[r][ht] += 1
        for tt in tail_types:
            rel_tail_types_all[r][tt] += 1

    relation_to_types: dict[str, tuple[str, str]] = {}
    relation_schema_source: dict[str, dict[str, str]] = {}
    for r in relations:
        rel_label = rel_meta.get(r, {}).get("label", r)
        rel_desc = rel_meta.get(r, {}).get("description", "")

        domain_candidates, range_candidates = _get_property_constraints(r)
        dom = _pick_best_type(domain_candidates, rel_head_types_all[r])
        rng = _pick_best_type(range_candidates, rel_tail_types_all[r])
        dom_src = "wikidata_constraint" if dom else "data_fallback"
        rng_src = "wikidata_constraint" if rng else "data_fallback"

        # Optional LLM fallback for missing side(s), if an API key is available.
        if llm_model and (dom is None or rng is None):
            llm_dom, llm_rng = _llm_fallback_domain_range(
                prop_id=r,
                prop_label=rel_label,
                prop_description=rel_desc,
                domain_candidates=domain_candidates,
                range_candidates=range_candidates,
                observed_head_types=rel_head_types_all[r],
                observed_tail_types=rel_tail_types_all[r],
                llm_model=llm_model,
            )
            if dom is None and llm_dom:
                dom = llm_dom
                dom_src = "llm_fallback"
            if rng is None and llm_rng:
                rng = llm_rng
                rng_src = "llm_fallback"

        # Final deterministic fallback: majority over all observed types for this relation side.
        if dom is None:
            dom = rel_head_types_all[r].most_common(1)[0][0] if rel_head_types_all[r] else "Entity"
            dom_src = "data_fallback"
        if rng is None:
            rng = rel_tail_types_all[r].most_common(1)[0][0] if rel_tail_types_all[r] else "Entity"
            rng_src = "data_fallback"

        # Coherence guard: keep only types that actually appear for this relation side.
        if dom != "Entity" and rel_head_types_all[r].get(dom, 0) <= 0:
            dom = rel_head_types_all[r].most_common(1)[0][0] if rel_head_types_all[r] else "Entity"
            dom_src = "data_fallback"
        if rng != "Entity" and rel_tail_types_all[r].get(rng, 0) <= 0:
            rng = rel_tail_types_all[r].most_common(1)[0][0] if rel_tail_types_all[r] else "Entity"
            rng_src = "data_fallback"

        # Low-confidence guard: if selected type matches too few triples, prefer generic Entity.
        n = max(1, int(rel_count[r]))
        dom_match = rel_head_types_all[r].get(dom, 0) / n if dom != "Entity" else 1.0
        rng_match = rel_tail_types_all[r].get(rng, 0) / n if rng != "Entity" else 1.0
        if dom != "Entity" and dom_match < float(min_type_match):
            dom = "Entity"
            dom_src = "low_confidence_entity"
        if rng != "Entity" and rng_match < float(min_type_match):
            rng = "Entity"
            rng_src = "low_confidence_entity"

        relation_to_types[r] = (dom, rng)
        relation_schema_source[r] = {"domain_source": dom_src, "range_source": rng_src}

    # Build schema store keyed by (domain, relation, range).
    pattern_heads: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    pattern_tails: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    pattern_support: Counter = Counter()
    for h, r, t in triples:
        dom, rng = relation_to_types[r]
        key = (dom, r, rng)
        pattern_heads[key].add(entity2id[h])
        pattern_tails[key].add(entity2id[t])
        pattern_support[key] += 1

    max_support = max(pattern_support.values()) if pattern_support else 1
    schema_store = {}
    for key, support in pattern_support.items():
        importance = 0.5 + 1.5 * (math.log1p(support) / math.log1p(max_support))
        schema_store[str(key)] = {
            "importance": round(float(importance), 4),
            "entities": {
                "head": sorted(pattern_heads[key]),
                "tail": sorted(pattern_tails[key]),
            },
            "relations": [key[1]],
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    # triples
    triples_lines = [f"{h}\t{r}\t{t}" for h, r, t in triples]
    (out_dir / "triples.txt").write_text("\n".join(triples_lines) + "\n", encoding="utf-8")

    # schema
    with (out_dir / "schema.txt").open("w", encoding="utf-8") as f:
        for r in relations:
            d, rg = relation_to_types[r]
            f.write(f"{d} {r} {rg}\n")

    # entity types
    entity_types_payload = {e: (entity2types.get(e, []) or [entity_primary_type[e]]) for e in entities}
    (out_dir / "entity_types.json").write_text(json.dumps(entity_types_payload, indent=2), encoding="utf-8")

    # mappings
    (out_dir / "entity_mappings.json").write_text(
        json.dumps(
            {
                "entity_to_id": entity2id,
                "id_to_entity": {str(i): e for i, e in id2entity.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    relation_schema = {
        r: {
            "domain": relation_to_types[r][0],
            "range": relation_to_types[r][1],
            "label": rel_meta.get(r, {}).get("label", r),
            "description": rel_meta.get(r, {}).get("description", ""),
            **relation_schema_source.get(r, {}),
        }
        for r in relations
    }

    # Human-readable schema descriptions (natural language).
    qids_for_labels = []
    for r in relations:
        dom, rng = relation_to_types[r]
        if isinstance(dom, str) and dom.startswith("Q"):
            qids_for_labels.append(dom)
        if isinstance(rng, str) and rng.startswith("Q"):
            qids_for_labels.append(rng)
    class_labels = _fetch_labels(list(dict.fromkeys(qids_for_labels)))

    schema_desc_json = {}
    schema_desc_lines: list[str] = []
    for r in relations:
        dom_id, rng_id = relation_to_types[r]
        dom_label = class_labels.get(dom_id, dom_id)
        rng_label = class_labels.get(rng_id, rng_id)
        rel_label = rel_meta.get(r, {}).get("label", r)
        rel_desc = rel_meta.get(r, {}).get("description", "")
        dom_src = relation_schema_source.get(r, {}).get("domain_source", "unknown")
        rng_src = relation_schema_source.get(r, {}).get("range_source", "unknown")
        sentence = (
            f"{r} ({rel_label}): expects a head entity of type '{dom_label}' "
            f"[{dom_id}] and a tail entity of type '{rng_label}' [{rng_id}]."
        )
        if rel_desc:
            sentence += f" Property meaning: {rel_desc}."
        sentence += f" Sources: domain={dom_src}, range={rng_src}."
        schema_desc_lines.append(sentence)
        schema_desc_json[r] = {
            "relation_id": r,
            "relation_label": rel_label,
            "relation_description": rel_desc,
            "domain": {"id": dom_id, "label": dom_label, "source": dom_src},
            "range": {"id": rng_id, "label": rng_label, "source": rng_src},
            "description_nl": sentence,
        }
    (out_dir / "relation_mappings.json").write_text(
        json.dumps(
            {
                "relation_to_id": relation2id,
                "id_to_relation": {str(i): r for i, r in id2relation.items()},
                "relation_to_types": relation_to_types,
                "relation_schema": relation_schema,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out_dir / "relation_names.txt").write_text("\n".join(relations) + "\n", encoding="utf-8")

    relation_to_pattern = {
        r: [relation_to_types[r][0], r, relation_to_types[r][1]]
        for r in relations
    }
    (out_dir / "relation_to_pattern.json").write_text(
        json.dumps(relation_to_pattern, indent=2), encoding="utf-8"
    )
    (out_dir / "schema_store.json").write_text(json.dumps(schema_store, indent=2), encoding="utf-8")
    (out_dir / "schema_descriptions.json").write_text(json.dumps(schema_desc_json, indent=2), encoding="utf-8")
    (out_dir / "schema_descriptions.txt").write_text("\n".join(schema_desc_lines) + "\n", encoding="utf-8")

    _write_timesteps(out_dir, triples_lines, steps=steps, percentage=percentage, seed=seed)

    metadata = (
        f"dataset: {out_dir.name}\n"
        f"source: CoDEx ({size.upper()})\n"
        f"source_repo: https://github.com/tsafavi/codex\n"
        f"entities: {len(entities)}\n"
        f"relations: {len(relations)}\n"
        f"triples_merged: {len(triples)}\n"
        f"timesteps_num: {steps}\n"
        f"per_step_percentage: {percentage}\n"
        f"seed: {seed}\n"
    )
    (out_dir / "metadata.yaml").write_text(metadata, encoding="utf-8")

    print(f"Prepared {out_dir.name}: entities={len(entities)} relations={len(relations)} triples={len(triples)}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and prepare CoDEx datasets.")
    p.add_argument(
        "--sizes",
        nargs="+",
        default=["s", "m", "l"],
        help="Subset sizes to prepare: s m l (default: all).",
    )
    p.add_argument("--steps", type=int, default=3, help="Number of timesteps.")
    p.add_argument("--percentage", type=int, default=20, help="Remove percentage per timestep.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--llm-model",
        default="gpt-5-mini",
        help=(
            "Optional LLM model for domain/range fallback when Wikidata constraints are missing. "
            "Used only if OPENAI_API_KEY is set."
        ),
    )
    p.add_argument(
        "--min-type-match",
        type=float,
        default=0.2,
        help="Minimum per-relation type match ratio; otherwise domain/range becomes Entity.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    root = Path(__file__).resolve().parent
    for s in args.sizes:
        out = root / f"CoDEx-{s.upper()}-{args.percentage}"
        prepare_codex_dataset(
            s,
            out,
            steps=args.steps,
            percentage=args.percentage,
            seed=args.seed,
            llm_model=args.llm_model,
            min_type_match=float(args.min_type_match),
        )
