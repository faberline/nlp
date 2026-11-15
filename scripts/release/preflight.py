#!/usr/bin/env python3
"""Fail closed before publishing artifacts."""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

root = pathlib.Path(__file__).resolve().parents[2]
tag, expected = os.environ['TAG_NAME'], os.environ['EXPECTED_SHA']
version = re.search(r'(?ms)^\[workspace\.package\]\n.*?^version\s*=\s*"([^"]+)"', (root / 'Cargo.toml').read_text()).group(1)
assert tag == 'v' + version, 'tag must equal workspace version'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
tag_sha = subprocess.check_output(['git', 'rev-parse', tag + '^{}'], cwd=root, text=True).strip()
assert head == expected == tag_sha, 'tag, workflow and checkout must name the same commit'
dist = pathlib.Path(sys.argv[1])
targets = ('aarch64-apple-darwin', 'x86_64-apple-darwin', 'x86_64-pc-windows-msvc')
checksums = []
for target in targets:
    ext = '.zip' if target.endswith('windows-msvc') else '.tar.gz'
    archive = dist / f'nlp-{version}-{target}{ext}'
    checksum = pathlib.Path(str(archive) + '.sha256')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert checksum.read_text(encoding='ascii').strip() == f'{digest}  {archive.name}'
    if ext == '.zip':
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.endswith('/build-info.json')]
            assert len(names) == 1
            info = json.loads(z.read(names[0]))
    else:
        with tarfile.open(archive, 'r:gz') as t:
            names = [n for n in t.getnames() if n.endswith('/build-info.json')]
            assert len(names) == 1
            info = json.load(t.extractfile(names[0]))
    assert info['version'] == version and info['source_commit'] == head and info['target'] == target and not info['models_included']
    checksums.append(f'{digest}  {archive.name}\n')
expected_names = {f'nlp-{version}-{t}' + ('.zip' if t.endswith('windows-msvc') else '.tar.gz') for t in targets}
assert {p.name for p in dist.iterdir()} == expected_names | {name + '.sha256' for name in expected_names}
(dist / 'SHA256SUMS').write_text(''.join(checksums), encoding='ascii')
url = f"https://api.github.com/repos/{os.environ['REPOSITORY']}/releases/tags/{urllib.parse.quote(tag, safe='')}"
request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json', 'User-Agent': 'nlp-release-preflight'})
try:
    with urllib.request.urlopen(request) as response:
        raise RuntimeError(f'release already exists: HTTP {response.status}')
except urllib.error.HTTPError as error:
    if error.code != 404: raise
base = f"https://api.github.com/repos/{os.environ['REPOSITORY']}/git/"
headers = {'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json', 'User-Agent': 'nlp-release-preflight'}
def github_json(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers)) as response:
        return json.load(response)
obj = github_json('ref/tags/' + urllib.parse.quote(tag, safe=''))['object']
for _ in range(8):
    if obj['type'] != 'tag':
        break
    obj = github_json('tags/' + obj['sha'])['object']
assert obj['type'] == 'commit' and obj['sha'] == head, 'remote tag moved or does not point to build commit'
print('release preflight passed')
