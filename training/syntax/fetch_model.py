#!/usr/bin/env python3
"""Pin `doc-syntax-v1` into `models/manifest.json` and optionally stage `models/doc-syntax-v1/`.

Keeps `models/test-tiny` untouched at `models[0]` (required by `tests/doc_model.rs`) and writes or
updates `doc-syntax-v1` at `models[1]` with exact SHA-256 hashes and byte sizes. Large model
binaries (`models/doc-syntax-v1/`) remain gitignored and out of git history.

Usage:
    # Pin SHA-256 hashes in models/manifest.json from staged export directory:
    training/.venv/bin/python training/syntax/fetch_model.py --stage-dir /tmp/nlp-models-stage --pin

    # Verify a staged or local models directory against models/manifest.json:
    training/.venv/bin/python training/syntax/fetch_model.py --verify-dir /tmp/nlp-models-stage

    # Copy staged files into local gitignored models/doc-syntax-v1/:
    training/.venv/bin/python training/syntax/fetch_model.py --stage-dir /tmp/nlp-models-stage --install
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "crates" / "faberline-nlp" / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
MODEL_ID = "doc-syntax-v1"
REQUIRED_FILES = ["model.onnx", "model.mlmodel", "tokenizer.json", "expected.json"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_entry_from_stage(stage_dir: Path, receipt_rel: str) -> dict:
    model_dir = stage_dir / MODEL_ID
    if not model_dir.is_dir():
        sys.exit(f"missing staged model directory: {model_dir}")
    files = []
    for name in REQUIRED_FILES:
        f = model_dir / name
        if not f.is_file():
            sys.exit(f"missing staged file: {f}")
        files.append(
            {
                "path": name,
                "sha256": sha256_of(f),
                "bytes": f.stat().st_size,
            }
        )
    tok_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab_len = len(tok_json["model"]["vocab"])
    exp_json = json.loads((model_dir / "expected.json").read_text(encoding="utf-8"))
    max_len = int(exp_json["max_len"])
    return {
        "id": MODEL_ID,
        "layers": ["pos", "syntax"],
        "licence": "MIT",
        "dir": MODEL_ID,
        "max_len": max_len,
        "pad_id": 0,
        "model": "model.onnx",
        "tokenizer": "tokenizer.json",
        "expected": "expected.json",
        "coreml": {"model": "model.mlmodel", "vocab": vocab_len},
        "files": files,
        "receipt": receipt_rel,
    }


def pin_manifest(stage_dir: Path, receipt_rel: str) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("schema") == "nlp.models.v1"
    models = [m for m in manifest["models"] if m["id"] != MODEL_ID]
    assert models and models[0]["id"] == "test-tiny", "models[0] must remain test-tiny"
    entry = build_entry_from_stage(stage_dir, receipt_rel)
    models.append(entry)
    manifest["models"] = models
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return entry


def verify_against_manifest(target_models_dir: Path) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = next((m for m in manifest["models"] if m["id"] == MODEL_ID), None)
    if entry is None:
        sys.exit(f"{MODEL_ID} is not pinned in {MANIFEST_PATH}")
    mdir = target_models_dir / entry["dir"]
    checked = []
    for f_spec in entry["files"]:
        fp = mdir / f_spec["path"]
        if not fp.is_file():
            sys.exit(f"missing {fp}")
        actual_bytes = fp.stat().st_size
        actual_sha = sha256_of(fp)
        if actual_bytes != f_spec["bytes"]:
            sys.exit(
                f"byte count mismatch for {fp}: expected {f_spec['bytes']}, got {actual_bytes}"
            )
        if actual_sha != f_spec["sha256"]:
            sys.exit(
                f"sha256 mismatch for {fp}: expected {f_spec['sha256']}, got {actual_sha}"
            )
        checked.append({"path": f_spec["path"], "bytes": actual_bytes, "sha256": actual_sha})
    return {"verified_dir": str(mdir), "files": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=Path("/tmp/nlp-models-stage"))
    parser.add_argument(
        "--receipt", type=str, default="docs/receipts/syntax-2026-11-11.json"
    )
    parser.add_argument("--pin", action="store_true", help="Pin doc-syntax-v1 in models/manifest.json")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy staged doc-syntax-v1 into local gitignored models/doc-syntax-v1/",
    )
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Verify doc-syntax-v1 files in <dir> against models/manifest.json",
    )
    args = parser.parse_args()

    out: dict = {}
    if args.pin:
        out["pinned"] = pin_manifest(args.stage_dir, args.receipt)
    if args.install:
        src = args.stage_dir / MODEL_ID
        dst = MODELS_DIR / MODEL_ID
        dst.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            shutil.copy2(src / name, dst / name)
        out["installed"] = str(dst)
    if args.verify_dir is not None:
        out["verified"] = verify_against_manifest(args.verify_dir)
    if not out:
        out["verified"] = verify_against_manifest(args.stage_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
