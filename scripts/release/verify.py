#!/usr/bin/env python3
"""Verify a packaged archive, then smoke the extracted executable."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

archive, checksum, target, expected_sha = map(str, sys.argv[1:5])
archive, checksum = pathlib.Path(archive), pathlib.Path(checksum)
line = checksum.read_text(encoding='ascii').strip()
digest, name = line.split('  ', 1)
assert name == archive.name and digest == hashlib.sha256(archive.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='nlp-archive-') as temp:
    dest = pathlib.Path(temp)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
            z.extractall(dest)
    else:
        with tarfile.open(archive, 'r:gz') as t:
            assert all(not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts and (m.isfile() or m.isdir()) for m in t.getmembers())
            t.extractall(dest)
    directories = list(dest.iterdir())
    assert len(directories) == 1
    root = directories[0]
    inventory = json.loads((root / 'inventory.json').read_text(encoding='utf-8'))
    actual = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p != root / 'inventory.json'}
    assert actual == inventory, 'archive inventory or file digest mismatch'
    meta = json.loads((root / 'build-info.json').read_text())
    source_root = pathlib.Path(__file__).resolve().parents[2]
    version = re.search(r'(?ms)^\[workspace\.package\]\n.*?^version\s*=\s*"([^"]+)"', (source_root / 'Cargo.toml').read_text()).group(1)
    assert meta['source_commit'] == expected_sha and meta['target'] == target and meta['version'] == version and not meta['models_included']
    assert (root / 'NOTICE.md').is_file() and (root / 'licenses/LICENSE-WORDNET.txt').is_file()
    executable = root / ('nlp.exe' if target.endswith('windows-msvc') else 'nlp')
    assert meta['binary_sha256'] == hashlib.sha256(executable.read_bytes()).hexdigest()
    receipt_path = root / 'licenses/dependency-notices.json'
    assert meta['notices_sha256'] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['target'] == target and receipt['features'] == meta['features']
    for package in receipt['closure']:
        for item in package['files']:
            assert actual['licenses/' + item['path']] == item['sha256']
    for item in receipt['tracked_lexicon_and_mamba_notices']:
        assert actual['licenses/' + item['path']] == item['sha256']
    assert meta['rustflags'] == ('-C target-feature=+crt-static' if target.endswith('windows-msvc') else '')
    assert meta['rustc_version'].startswith('rustc ')
    allowed_top = {executable.name, 'LICENSE', 'NOTICE.md', 'runtime-dependencies.txt', 'build-info.json', 'inventory.json', 'licenses'}
    assert {p.name for p in root.iterdir()} == allowed_top
    assert all(not p.name.lower().endswith(('.exe', '.dll', '.dylib', '.so')) for p in (root / 'licenses').rglob('*') if p.is_file())
    subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('smoke.py')), str(executable), *sys.argv[5:]], check=True)
