#!/usr/bin/env python3
"""Select top 1,500 VERB and top 1,000 MULTI Japanese lexicon rows ranked by SudachiDict cost.

Reads `crates/faberline-nlp/lexicon/data/ja.tsv` and SudachiDict raw CSVs
(`small_lex.csv` + `core_lex.csv`) from `--sudachi-csv-dir`.

Outputs:
- `tools/lexicon_build/data/a4_sudachi_ranks.tsv` (2,500 rows; no Sudachi CSV glosses)
- Optional 500-row author input shards (`input_shard_{1..5}.tsv`) when `--shards-dir` is given.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JA_TSV = REPO_ROOT / "crates/faberline-nlp/lexicon/data/ja.tsv"
DEFAULT_MANIFEST = REPO_ROOT / "tools/lexicon_build/data/a4_sudachi_ranks.tsv"

EXCLUDED_MULTI_POS1 = frozenset({"補助記号", "記号", "空白"})
PURE_KATA_RE = re.compile(r"^[\u30A0-\u30FFー]+$")


def map_sudachi_pos_to_upos(pos1: str, pos2: str, pos3: str) -> frozenset[str]:
    if pos1 == "名詞":
        if pos2 == "数詞":
            return frozenset({"NUM"})
        upos = {"NOUN"}
        if "サ変" in pos3:
            upos.add("VERB")
        if "副詞可能" in pos3:
            upos.add("ADV")
        if "形状詞可能" in pos3:
            upos.add("ADJ")
        return frozenset(upos)
    if pos1 == "代名詞":
        return frozenset({"PRON"})
    if pos1 == "動詞":
        return frozenset({"VERB", "AUX"}) if pos2 == "非自立可能" else frozenset({"VERB"})
    if pos1 in {"形容詞", "形状詞", "連体詞"}:
        return frozenset({"ADJ"})
    if pos1 == "副詞":
        return frozenset({"ADV"})
    if pos1 == "接続詞":
        return frozenset({"CCONJ", "SCONJ"})
    if pos1 == "感動詞":
        return frozenset({"INTJ"})
    if pos1 == "助詞":
        return frozenset({"ADP", "PART", "CCONJ", "SCONJ"})
    if pos1 == "助動詞":
        return frozenset({"AUX"})
    if pos1 in {"接頭辞", "接尾辞"}:
        return frozenset({"NOUN", "NUM", "ADJ", "VERB"})
    return frozenset()


def load_sudachi_costs(sudachi_dir: Path) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, list[tuple[frozenset[str], int]]],
    dict[str, int],
]:
    """Load Sudachi costs for VERB (終止形 primary + fallbacks), MULTI (surface -> [(upos, cost)]), and PROPN."""
    verb_end_cost: dict[str, int] = {}
    verb_any_cost: dict[str, int] = {}
    verb_norm_end_cost: dict[str, int] = {}
    multi_surface_entries: dict[str, dict[frozenset[str], int]] = defaultdict(dict)
    propn_min_cost: dict[str, int] = {}

    for fname in ("small_lex.csv", "core_lex.csv"):
        csv_path = sudachi_dir / fname
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing Sudachi CSV: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 13:
                    continue
                try:
                    left_id = int(row[1])
                    cost = int(row[3])
                except ValueError:
                    continue
                if left_id < 0:
                    continue

                surface = row[0]
                pos1 = row[5]
                pos2 = row[6]
                pos3 = row[7]
                cform = row[10]
                norm = row[12]

                if pos2 == "固有名詞":
                    prev_p = propn_min_cost.get(surface)
                    if prev_p is None or cost < prev_p:
                        propn_min_cost[surface] = cost
                    continue

                if pos1 in EXCLUDED_MULTI_POS1:
                    continue

                if pos1 == "動詞":
                    prev_a = verb_any_cost.get(surface)
                    if prev_a is None or cost < prev_a:
                        verb_any_cost[surface] = cost
                    if cform.startswith("終止形"):
                        prev_e = verb_end_cost.get(surface)
                        if prev_e is None or cost < prev_e:
                            verb_end_cost[surface] = cost
                        if norm:
                            prev_ne = verb_norm_end_cost.get(norm)
                            if prev_ne is None or cost < prev_ne:
                                verb_norm_end_cost[norm] = cost

                upos_set = map_sudachi_pos_to_upos(pos1, pos2, pos3)
                if upos_set:
                    by_upos = multi_surface_entries[surface]
                    prev_m = by_upos.get(upos_set)
                    if prev_m is None or cost < prev_m:
                        by_upos[upos_set] = cost

    multi_list = {surf: list(by_upos.items()) for surf, by_upos in multi_surface_entries.items()}
    return verb_end_cost, verb_any_cost, verb_norm_end_cost, multi_list, propn_min_cost


def select_ranks(
    ja_tsv_path: Path,
    verb_end_cost: dict[str, int],
    verb_any_cost: dict[str, int],
    verb_norm_end_cost: dict[str, int],
    multi_entries: dict[str, list[tuple[frozenset[str], int]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    verb_candidates: list[dict[str, object]] = []
    multi_candidates: list[dict[str, object]] = []

    with ja_tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            cols = raw.split("\t")
            if len(cols) < 8:
                continue
            surface, variant, ja_cost_str, decided_upos, upos_mask, lemma, counterpart, meta = cols[:8]
            if counterpart != "-":
                continue
            ja_cost = int(ja_cost_str)

            if decided_upos == "VERB":
                scost = verb_end_cost.get(surface)
                if scost is None:
                    scost = verb_any_cost.get(surface)
                if scost is None:
                    scost = verb_norm_end_cost.get(surface)
                if scost is not None:
                    verb_candidates.append(
                        {
                            "group": "VERB",
                            "surface": surface,
                            "variant": variant,
                            "ja_cost": ja_cost,
                            "decided_upos": decided_upos,
                            "upos_mask": upos_mask,
                            "lemma": lemma,
                            "sudachi_cost": scost,
                            "meta": meta,
                        }
                    )
            elif decided_upos == "-":
                mask_parts = upos_mask.split("|")
                if "PROPN" in mask_parts:
                    continue
                mask_set = frozenset(mask_parts)
                entries = multi_entries.get(surface)
                if not entries:
                    continue
                matched_costs = [c for u_set, c in entries if u_set & mask_set]
                if matched_costs:
                    multi_candidates.append(
                        {
                            "group": "MULTI",
                            "surface": surface,
                            "variant": variant,
                            "ja_cost": ja_cost,
                            "decided_upos": decided_upos,
                            "upos_mask": upos_mask,
                            "lemma": lemma,
                            "sudachi_cost": min(matched_costs),
                            "meta": meta,
                        }
                    )

    sort_key = lambda r: (
        int(r["sudachi_cost"]),  # type: ignore[arg-type]
        int(r["ja_cost"]),  # type: ignore[arg-type]
        str(r["surface"]).encode("utf-8"),
        str(r["variant"]).encode("utf-8"),
    )
    verb_candidates.sort(key=sort_key)
    multi_candidates.sort(key=sort_key)

    if len(verb_candidates) < 1500:
        raise RuntimeError(f"Only {len(verb_candidates)} VERB candidates matched (need 1500)")
    if len(multi_candidates) < 1000:
        raise RuntimeError(f"Only {len(multi_candidates)} MULTI candidates matched (need 1000)")

    verb_top = verb_candidates[:1500]
    multi_top = multi_candidates[:1000]
    for idx, row in enumerate(verb_top, start=1):
        row["rank"] = idx
    for idx, row in enumerate(multi_top, start=1):
        row["rank"] = idx

    return verb_top, multi_top


def write_manifest(out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("group\trank\tsurface\tvariant\tdecided_upos\tupos_mask\tlemma\tsudachi_cost\n")
        for r in rows:
            f.write(
                f"{r['group']}\t{r['rank']}\t{r['surface']}\t{r['variant']}\t"
                f"{r['decided_upos']}\t{r['upos_mask']}\t{r['lemma']}\t{r['sudachi_cost']}\n"
            )


def write_shards(shards_dir: Path, verb_top: list[dict[str, object]], multi_top: list[dict[str, object]]) -> None:
    shards_dir.mkdir(parents=True, exist_ok=True)
    slices = [
        ("input_shard_1.tsv", verb_top[0:500]),
        ("input_shard_2.tsv", verb_top[500:1000]),
        ("input_shard_3.tsv", verb_top[1000:1500]),
        ("input_shard_4.tsv", multi_top[0:500]),
        ("input_shard_5.tsv", multi_top[500:1000]),
    ]
    header = "group\trank\tsurface\tvariant\tdecided_upos\tupos_mask\tlemma\tja_cost\tsudachi_cost\tmeta\n"
    for fname, subset in slices:
        p = shards_dir / fname
        with p.open("w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            for r in subset:
                f.write(
                    f"{r['group']}\t{r['rank']}\t{r['surface']}\t{r['variant']}\t"
                    f"{r['decided_upos']}\t{r['upos_mask']}\t{r['lemma']}\t"
                    f"{r['ja_cost']}\t{r['sudachi_cost']}\t{r['meta']}\n"
                )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sudachi-csv-dir",
        type=Path,
        required=True,
        help="Directory containing SudachiDict small_lex.csv and core_lex.csv",
    )
    parser.add_argument(
        "--ja-tsv",
        type=Path,
        default=DEFAULT_JA_TSV,
        help="Path to ja.tsv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Output manifest TSV path",
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="Optional output directory for 500-row author input shards",
    )
    args = parser.parse_args(argv)

    verb_end, verb_any, verb_norm_end, multi_entries, propn_min = load_sudachi_costs(args.sudachi_csv_dir)
    verb_top, multi_top = select_ranks(args.ja_tsv, verb_end, verb_any, verb_norm_end, multi_entries)
    write_manifest(args.out_manifest, verb_top + multi_top)
    if args.shards_dir is not None:
        write_shards(args.shards_dir, verb_top, multi_top)

    print(
        f"VERB 1..1500: rank1={verb_top[0]['surface']}({verb_top[0]['sudachi_cost']}), "
        f"rank1500={verb_top[-1]['surface']}({verb_top[-1]['sudachi_cost']})"
    )
    print(
        f"MULTI 1..1000: rank1={multi_top[0]['surface']}({multi_top[0]['sudachi_cost']}), "
        f"rank1000={multi_top[-1]['surface']}({multi_top[-1]['sudachi_cost']})"
    )
    mask_counts = Counter(str(r["upos_mask"]) for r in multi_top)
    print("MULTI upos_mask distribution:", dict(mask_counts.most_common()))

    propn_cheaper = []
    for r in multi_top:
        surf = str(r["surface"])
        if PURE_KATA_RE.match(surf):
            pcost = propn_min.get(surf)
            if pcost is not None and pcost < int(r["sudachi_cost"]):  # type: ignore[arg-type]
                propn_cheaper.append((int(r["rank"]), surf, int(r["sudachi_cost"]), pcost))  # type: ignore[arg-type]
    print(f"MULTI pure-katakana with lower 固有名詞 cost ({len(propn_cheaper)}):", propn_cheaper[:25])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
