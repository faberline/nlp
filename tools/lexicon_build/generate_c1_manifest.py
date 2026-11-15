#!/usr/bin/env python3
"""Deterministic row-by-row physical reconciliation manifest for C1 lexicon builds."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

BASE_REF_DEFAULT = "d99dd7b68001687b3cd76023ac32785abdc32f06"
LEXICON_FILES = {
    "en": "crates/faberline-nlp/lexicon/data/en.tsv",
    "zh_hant": "crates/faberline-nlp/lexicon/data/zh_hant.tsv",
    "zh_hans": "crates/faberline-nlp/lexicon/data/zh_hans.tsv",
}
VARIANT_ORDER = {
    "zh-Hant": 0,
    "zh-Hans": 1,
    "en-US": 2,
    "en-GB": 3,
    "en": 4,
    "ja": 5,
    "ko": 6,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_physical_rows(text: str) -> list[tuple[int, str, str, int, str, str, str, str, str]]:
    rows = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        rows.append((
            idx,
            p[0],
            p[1],
            int(p[2]),
            p[3],
            p[4],
            p[5],
            p[6],
            p[7] if len(p) > 7 else "-",
        ))
    return rows


def merge_entries_lookup(rows_list: list[tuple[int, str, str, int, str, str, str, str, str]]) -> str:
    """Simulate build.rs sort + src/lexicon/mod.rs merge_entries effective lookup."""
    if not rows_list:
        return "-"
    ordered = sorted(rows_list, key=lambda r: (VARIANT_ORDER.get(r[2], 99), r[3]))
    min_cost = min(r[3] for r in ordered)
    mask_parts: set[str] = set()
    decided_candidates: set[str] = set()
    lemma = "-"
    counterpart = "-"
    sem_classes: set[str] = set()
    for r in ordered:
        if r[4] != "-":
            decided_candidates.add(r[4])
        if r[5] != "-":
            for tag in r[5].split("|"):
                mask_parts.add(tag)
        if lemma == "-" and r[6] != "-":
            lemma = r[6]
        if counterpart == "-" and r[7] != "-":
            counterpart = r[7]
        if r[8].startswith("sem="):
            sem_classes.add(r[8])
    decided = next(iter(decided_candidates)) if len(decided_candidates) == 1 and len(mask_parts) == 1 else "-"
    mask = "|".join(sorted(mask_parts)) if mask_parts else "-"
    meta = next(iter(sem_classes)) if len(sem_classes) == 1 else "-"
    return f"{min_cost}/{decided}/{mask}/{lemma}/{counterpart}/{meta}"


def build_group_lookup(
    files: dict[str, list[tuple[int, str, str, int, str, str, str, str, str]]]
) -> dict[str, str]:
    by_surface: dict[str, list] = {}
    for rows in files.values():
        for r in rows:
            by_surface.setdefault(r[1], []).append(r)
    return {surf: merge_entries_lookup(rlist) for surf, rlist in by_surface.items()}


def git_show_text(repo_root: Path, ref: str, rel_path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{rel_path}"], cwd=repo_root, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physical-row C1 impact manifests.")
    parser.add_argument("--base-ref", default=BASE_REF_DEFAULT, help="Base git ref to compare against")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for manifests")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_texts = {k: git_show_text(repo_root, args.base_ref, p) for k, p in LEXICON_FILES.items()}
    new_texts = {k: (repo_root / p).read_text(encoding="utf-8") for k, p in LEXICON_FILES.items()}

    old_rows_map = {k: parse_physical_rows(v) for k, v in old_texts.items()}
    new_rows_map = {k: parse_physical_rows(v) for k, v in new_texts.items()}

    # Cross-script LangGroup::Zh lookup + EN lookup
    old_zh_lookup = build_group_lookup({"zh_hant": old_rows_map["zh_hant"], "zh_hans": old_rows_map["zh_hans"]})
    new_zh_lookup = build_group_lookup({"zh_hant": new_rows_map["zh_hant"], "zh_hans": new_rows_map["zh_hans"]})
    old_en_lookup = build_group_lookup({"en": old_rows_map["en"]})
    new_en_lookup = build_group_lookup({"en": new_rows_map["en"]})

    summary: dict[str, object] = {
        "base_ref": args.base_ref,
        "files": {},
        "removed_duplicates": [],
        "field_changes": [],
    }

    for lang_key, rel_path in LEXICON_FILES.items():
        old_rows = old_rows_map[lang_key]
        new_rows = new_rows_map[lang_key]
        old_eff_map = old_en_lookup if lang_key == "en" else old_zh_lookup
        new_eff_map = new_en_lookup if lang_key == "en" else new_zh_lookup

        old_by_key: dict[tuple[str, str], list] = {}
        for r in old_rows:
            old_by_key.setdefault((r[1], r[2]), []).append(r)
        new_by_key: dict[tuple[str, str], list] = {}
        for r in new_rows:
            new_by_key.setdefault((r[1], r[2]), []).append(r)

        all_keys = sorted(set(old_by_key.keys()) | set(new_by_key.keys()))
        counts: Counter = Counter()
        deltas: list[int] = []
        dup_extra = 0

        manifest_path = out_dir / f"manifest_{lang_key}.tsv"
        with manifest_path.open("w", encoding="utf-8") as mf:
            mf.write(
                "surface\tvariant\tchange_category\told_line\tnew_line\told_cost\tnew_cost\tdelta_cost\t"
                "old_fields\tnew_fields\told_effective_lookup\tnew_effective_lookup\n"
            )
            for k in all_keys:
                o_list = old_by_key.get(k, [])
                n_list = new_by_key.get(k, [])
                if len(o_list) > 1:
                    dup_extra += len(o_list) - 1
                    summary["removed_duplicates"].append({
                        "file": lang_key,
                        "surface": k[0],
                        "variant": k[1],
                        "old_rows": [{"line": r[0], "cost": r[3], "fields": "|".join(r[4:])} for r in o_list],
                        "new_rows": [{"line": r[0], "cost": r[3], "fields": "|".join(r[4:])} for r in n_list],
                    })
                m = max(len(o_list), len(n_list))
                old_eff = old_eff_map.get(k[0], "-")
                new_eff = new_eff_map.get(k[0], "-")
                for i in range(m):
                    o = o_list[i] if i < len(o_list) else None
                    n = n_list[i] if i < len(n_list) else None
                    if o and n:
                        dc = n[3] - o[3]
                        deltas.append(dc)
                        o_f = "\t".join(o[4:])
                        n_f = "\t".join(n[4:])
                        if dc == 0 and o_f == n_f:
                            cat = "UNCHANGED"
                        elif dc != 0 and o_f == n_f:
                            cat = "COST_ONLY"
                        elif dc == 0 and o_f != n_f:
                            cat = "FIELDS_ONLY"
                        else:
                            cat = "COST_AND_FIELDS"
                        counts[cat] += 1
                        if cat in ("FIELDS_ONLY", "COST_AND_FIELDS"):
                            summary["field_changes"].append({
                                "file": lang_key,
                                "surface": k[0],
                                "variant": k[1],
                                "category": cat,
                                "delta_cost": dc,
                                "old_fields": "|".join(o[4:]),
                                "new_fields": "|".join(n[4:]),
                            })
                        mf.write(
                            f"{k[0]}\t{k[1]}\t{cat}\t{o[0]}\t{n[0]}\t{o[3]}\t{n[3]}\t{dc}\t"
                            f"{'|'.join(o[4:])}\t{'|'.join(n[4:])}\t{old_eff}\t{new_eff}\n"
                        )
                    elif o and not n:
                        cat = "REMOVED_DUPLICATE" if i > 0 else "REMOVED"
                        counts[cat] += 1
                        mf.write(
                            f"{k[0]}\t{k[1]}\t{cat}\t{o[0]}\t-\t{o[3]}\t-\t-\t"
                            f"{'|'.join(o[4:])}\t-\t{old_eff}\t{new_eff}\n"
                        )
                    else:
                        cat = "ADDED"
                        counts[cat] += 1
                        mf.write(
                            f"{k[0]}\t{k[1]}\t{cat}\t-\t{n[0]}\t-\t{n[3]}\t-\t"
                            f"-\t{'|'.join(n[4:])}\t{old_eff}\t{new_eff}\n"
                        )

        deltas.sort()
        summary["files"][lang_key] = {
            "path": rel_path,
            "old_physical_rows": len(old_rows),
            "new_physical_rows": len(new_rows),
            "old_unique_keys": len(old_by_key),
            "new_unique_keys": len(new_by_key),
            "old_extra_duplicate_rows": dup_extra,
            "reconciled_old_check": sum(counts[c] for c in counts if c != "ADDED"),
            "reconciled_new_check": sum(counts[c] for c in counts if not c.startswith("REMOVED")),
            "new_bytes": len(new_texts[lang_key].encode("utf-8")),
            "new_sha256": sha256_bytes(new_texts[lang_key].encode("utf-8")),
            "change_counts": dict(counts),
            "delta_cost_min": deltas[0] if deltas else None,
            "delta_cost_median": deltas[len(deltas) // 2] if deltas else None,
            "delta_cost_max": deltas[-1] if deltas else None,
            "manifest_path": manifest_path.name,
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        }

    summary_path = out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "files": summary["files"],
        "removed_duplicates_count": len(summary["removed_duplicates"]),
        "field_changes_count": len(summary["field_changes"]),
        "summary_sha256": sha256_bytes(summary_path.read_bytes()),
    }, indent=2))


if __name__ == "__main__":
    main()
