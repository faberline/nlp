#!/usr/bin/env python3
"""Package one already-built executable and its tracked notices."""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

binary, target, destination, metadata_path = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
runtime_report, rustflags = pathlib.Path(sys.argv[5]), sys.argv[6]
root = pathlib.Path(__file__).resolve().parents[2]
version = re.search(r'(?ms)^\[workspace\.package\]\n.*?^version\s*=\s*"([^"]+)"', (root / 'Cargo.toml').read_text()).group(1)
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert len(sha) == 40 and binary.is_file()
destination.mkdir(parents=True, exist_ok=True)
stem = f'nlp-{version}-{target}'
with tempfile.TemporaryDirectory() as tmp:
    stage = pathlib.Path(tmp) / stem
    stage.mkdir()
    shutil.copy2(binary, stage / binary.name)
    shutil.copy2(root / 'LICENSE', stage / 'LICENSE')
    shutil.copy2(runtime_report, stage / 'runtime-dependencies.txt')
    notices = stage / 'licenses'
    shutil.copytree(root / 'crates/faberline-nlp/lexicon/LICENSES', notices)
    shutil.copy2(root / 'crates/faberline-nlp/LICENSE-MAMBA', notices / 'LICENSE-MAMBA')
    subprocess.run([sys.executable, str(root / 'scripts/release/collect_notices.py'), str(metadata_path), target, str(notices)], check=True)
    shutil.copy2(root / 'docs/releases/NOTICE.md', stage / 'NOTICE.md')
    metadata = {'version': version, 'source_commit': sha, 'target': target, 'product': 'cpu-rules-and-embedded-dictionary', 'models_included': False, 'features': 'workspace default CPU features', 'rustflags': rustflags, 'rustc_version': subprocess.check_output(['rustc', '-vV'], text=True).strip(), 'binary_sha256': hashlib.sha256((stage / binary.name).read_bytes()).hexdigest(), 'notices_sha256': hashlib.sha256((notices / 'dependency-notices.json').read_bytes()).hexdigest()}
    (stage / 'build-info.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    inventory = {str(p.relative_to(stage)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage / 'inventory.json').write_text(json.dumps(inventory, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    if target.endswith('windows-msvc'):
        archive = destination / (stem + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file(): z.write(p, p.relative_to(stage.parent))
    else:
        archive = destination / (stem + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar: tar.add(stage, arcname=stem)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (destination / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    print(archive)
