#!/usr/bin/env python3
"""Prepare resumable silver-label batches from raw text or JSONL using nlp_doc (layers 1 and 2).

Issue #27: slices input items into fixed-size batches tokenized with the repository's real
Layer 1/2 rules (`nlp_doc`), writing `<work-dir>/manifest.json` (`nlp.silver.v1`) with
`source_sha256`, `model_id` (`claude-sonnet-5-5`), and `prompt_version`.

Uses only the Python standard library and runs a local `nlp_doc` binary. Never calls a network
API and never embeds API keys. Re-running on an existing work directory preserves completed
Pass-A / Pass-B / merged batch files and refuses mismatched `source_sha256`, `model_id`,
`prompt_version`, or `batch_size` unless `--force` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "nlp.silver.v1"
DEFAULT_MODEL_ID = "claude-sonnet-5-5"
DEFAULT_PROMPT_VERSION = "silver-v1"
DEFAULT_BATCH_SIZE = 50


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_raw_items(input_path: Path, raw_bytes: bytes) -> list[dict]:
    text = raw_bytes.decode("utf-8")
    if input_path.suffix.lower() == ".jsonl":
        items = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                sys.exit(f"{input_path}:{lineno}: invalid JSON: {exc}")
            if not isinstance(obj, dict) or "text" not in obj or not isinstance(obj["text"], str):
                sys.exit(f"{input_path}:{lineno}: expected JSON object with string 'text'")
            item_id = str(obj.get("id", f"item_{lineno:06d}"))
            items.append(
                {
                    "id": item_id,
                    "genre": str(obj.get("genre", "unspecified")),
                    "lang": str(obj.get("lang", "unspecified")),
                    "text": obj["text"],
                }
            )
        return items

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        {
            "id": f"item_{idx:06d}",
            "genre": "unspecified",
            "lang": "unspecified",
            "text": para,
        }
        for idx, para in enumerate(paragraphs, start=1)
    ]


def tokenize_with_nlp_doc(binary: Path, text: str) -> tuple[list[str], list[str | None], list[list[int]]]:
    proc = subprocess.run(
        [str(binary)],
        input=text.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(f"nlp_doc failed (exit {proc.returncode}): {err}")
    doc = json.loads(proc.stdout.decode("utf-8"))
    toks = doc.get("tokens", [])
    if not toks:
        return [], [], []
    sents_by_idx: dict[int, list[int]] = {}
    for idx, tok in enumerate(toks):
        sents_by_idx.setdefault(int(tok["sentence"]), []).append(idx)
    tokens = [text[int(t["span"]["start_char"]) : int(t["span"]["end_char"])] for t in toks]
    upos = [t.get("upos") for t in toks]
    sentences = [[indices[0], indices[-1] + 1] for _, indices in sorted(sents_by_idx.items())]
    return tokens, upos, sentences


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(content, encoding="utf-8")


def prepare_batches(
    input_path: Path,
    work_dir: Path,
    nlp_doc: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    force: bool = False,
) -> dict:
    if batch_size <= 0:
        sys.exit("--batch-size must be positive")
    if not input_path.is_file():
        sys.exit(f"input file not found: {input_path}")
    if not nlp_doc.is_file():
        sys.exit(f"nlp_doc binary not found: {nlp_doc}")

    raw_bytes = input_path.read_bytes()
    source_sha = sha256_bytes(raw_bytes)

    manifest_path = work_dir / "manifest.json"
    existing_batches: dict[str, dict] = {}
    if manifest_path.exists() and not force:
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prev.get("schema") != SCHEMA:
            sys.exit(f"existing manifest has schema {prev.get('schema')!r}, expected {SCHEMA!r}")
        for key, want in [
            ("source_sha256", source_sha),
            ("model_id", model_id),
            ("prompt_version", prompt_version),
            ("batch_size", batch_size),
        ]:
            if prev.get(key) != want:
                sys.exit(
                    f"resume mismatch on {key}: manifest has {prev.get(key)!r}, got {want!r} "
                    "(pass --force to start fresh)"
                )
        for b in prev.get("batches", []):
            existing_batches[b["batch_id"]] = b

    raw_items = load_raw_items(input_path, raw_bytes)
    prepared: list[dict] = []
    skipped_empty = 0

    # If every batch input file already exists with matching sha256 on resume, skip re-running nlp_doc.
    batches_dir = work_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    can_fast_resume = bool(existing_batches) and all(
        (work_dir / b["input_file"]).is_file()
        and sha256_file(work_dir / b["input_file"]) == b["input_sha256"]
        for b in existing_batches.values()
    )
    if can_fast_resume:
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        return prev

    for item in raw_items:
        tokens, layer2_upos, sentences = tokenize_with_nlp_doc(nlp_doc, item["text"])
        if not tokens:
            skipped_empty += 1
            continue
        prepared.append(
            {
                "id": item["id"],
                "genre": item["genre"],
                "lang": item["lang"],
                "text": item["text"],
                "tokens": tokens,
                "layer2_upos": layer2_upos,
                "sentences": sentences,
            }
        )

    batches_meta: list[dict] = []
    for batch_idx, start in enumerate(range(0, len(prepared), batch_size)):
        chunk = prepared[start : start + batch_size]
        batch_id = f"batch_{batch_idx:04d}"
        rel_in = f"batches/{batch_id}.input.jsonl"
        rel_a = f"batches/{batch_id}.pass_a.jsonl"
        rel_b = f"batches/{batch_id}.pass_b.jsonl"
        rel_merged = f"batches/{batch_id}.merged.jsonl"

        in_path = work_dir / rel_in
        write_jsonl(in_path, chunk)
        in_sha = sha256_file(in_path)

        prev_b = existing_batches.get(batch_id)
        status = "pending"
        counts = None
        if prev_b and prev_b.get("input_sha256") == in_sha:
            status = prev_b.get("status", "pending")
            counts = prev_b.get("counts")
        else:
            has_a = (work_dir / rel_a).is_file()
            has_b = (work_dir / rel_b).is_file()
            if has_a and has_b:
                status = "ready_to_merge"
            elif has_a or has_b:
                status = "partial"

        batches_meta.append(
            {
                "batch_id": batch_id,
                "items": len(chunk),
                "input_file": rel_in,
                "input_sha256": in_sha,
                "pass_a_file": rel_a,
                "pass_b_file": rel_b,
                "merged_file": rel_merged,
                "status": status,
                "counts": counts,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "source_path": str(input_path),
        "source_sha256": source_sha,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "batch_size": batch_size,
        "total_items": len(prepared),
        "skipped_empty": skipped_empty,
        "batches": batches_meta,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw UTF-8 text or JSONL file")
    parser.add_argument("--work-dir", required=True, type=Path, help="Resumable silver work directory")
    parser.add_argument("--nlp-doc", required=True, type=Path, help="Path to local nlp_doc binary")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--force", action="store_true", help="Overwrite manifest on parameter change")
    args = parser.parse_args()

    manifest = prepare_batches(
        input_path=args.input,
        work_dir=args.work_dir,
        nlp_doc=args.nlp_doc,
        batch_size=args.batch_size,
        model_id=args.model_id,
        prompt_version=args.prompt_version,
        force=args.force,
    )
    print(
        f"prepared {manifest['total_items']} items in {len(manifest['batches'])} batches "
        f"(skipped_empty={manifest['skipped_empty']}, source_sha256={manifest['source_sha256'][:12]}...)"
    )


if __name__ == "__main__":
    main()
