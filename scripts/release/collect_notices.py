#!/usr/bin/env python3
"""Collect locked dependency notices from Cargo metadata and record their hashes."""
import hashlib
import json
import pathlib
import re
import shutil
import sys

metadata_path, target, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
root = pathlib.Path(__file__).resolve().parents[2]
lock_bytes = (root / 'Cargo.lock').read_bytes()
lock = {}
for section in lock_bytes.decode('utf-8').split('[[package]]')[1:]:
    def field(name):
        match = re.search(r'^' + name + r' = "([^"]+)"$', section, re.MULTILINE)
        return match.group(1) if match else None
    lock[(field('name'), field('version'), field('source'))] = field('checksum')
packages = {p['id']: p for p in metadata['packages']}
nodes = {n['id']: n for n in metadata['resolve']['nodes']}
roots = [p['id'] for p in packages.values() if p['name'] == 'faberline-nlp-cli' and p['source'] is None]
assert len(roots) == 1
todo, seen = roots[:], set()
while todo:
    ident = todo.pop()
    if ident in seen:
        continue
    seen.add(ident)
    for dependency in nodes[ident]['deps']:
        if any(k['kind'] != 'dev' for k in dependency['dep_kinds']):
            todo.append(dependency['pkg'])

out.mkdir(parents=True, exist_ok=True)
entries = []
for ident in sorted(seen):
    package = packages[ident]
    source_dir = pathlib.Path(package['manifest_path']).parent
    entry = {'name': package['name'], 'version': package['version'], 'source': package['source'],
             'features': nodes[ident]['features'], 'license_expression': package['license'], 'license_file': package['license_file'], 'files': [],
             'cargo_lock_checksum': lock.get((package['name'], package['version'], package['source'])),
             'manifest_sha256': hashlib.sha256((source_dir / 'Cargo.toml').read_bytes()).hexdigest()}
    vcs = source_dir / '.cargo_vcs_info.json'
    if vcs.is_file():
        entry['vcs'] = json.loads(vcs.read_text(encoding='utf-8'))
    # Workspace license notices are collected separately. Never infer a copyright owner.
    if package['source'] is not None:
        candidates = {p for p in source_dir.iterdir() if p.is_file() and p.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE'))}
        if package['license_file']:
            candidates.add(source_dir / package['license_file'])
        for path in sorted(candidates):
            assert path.is_file() and path.resolve().is_relative_to(source_dir.resolve()), f'invalid notice path: {path}'
            dest = out / 'dependencies' / (package['name'] + '-' + package['version']) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            entry['files'].append({'path': str(dest.relative_to(out)).replace('\\', '/'), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
        recovered = {
            ('include-flate-codegen', '0.3.5'): ('67e17da7efc43f3bd92dae45b43c3baa1c543c9e', root / 'scripts/release/licenses/LICENSE-INCLUDE-FLATE-APACHE-2.0.txt', 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
            ('include-flate-compress', '0.3.5'): ('67e17da7efc43f3bd92dae45b43c3baa1c543c9e', root / 'scripts/release/licenses/LICENSE-INCLUDE-FLATE-APACHE-2.0.txt', 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
            ('jieba-rs', '0.11.0'): ('3a0d75cf7455e2006330a5b4e94156da272cd7b9', root / 'crates/faberline-nlp/lexicon/LICENSES/LICENSE-JIEBA-MIT.txt', '92257c8d1c7a059fa02720b34fce017574a990bed14cee8f565597f4fbfb1f13'),
            ('jieba-macros', '0.11.0'): ('3a0d75cf7455e2006330a5b4e94156da272cd7b9', root / 'crates/faberline-nlp/lexicon/LICENSES/LICENSE-JIEBA-MIT.txt', '92257c8d1c7a059fa02720b34fce017574a990bed14cee8f565597f4fbfb1f13'),
        }
        key = (package['name'], package['version'])
        if not entry['files'] and key in recovered:
            revision, path, digest = recovered[key]
            assert entry.get('vcs', {}).get('git', {}).get('sha1') == revision
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual_digest == digest, (
                f'recovered notice mismatch: {package["name"]} {package["version"]} '
                f'{path.name}: expected {digest}, actual {actual_digest}'
            )
            dest = out / 'dependencies' / (package['name'] + '-' + package['version']) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            entry['files'].append({'path': str(dest.relative_to(out)).replace('\\', '/'), 'sha256': digest, 'recovered_upstream_revision': revision})
    entries.append(entry)
tracked = [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'dependency-notices.json']
receipt = {'target': target, 'features': 'workspace default CPU features', 'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(), 'cargo_lock_sha256': hashlib.sha256(lock_bytes).hexdigest(), 'closure': entries, 'tracked_lexicon_and_mamba_notices': tracked,
           'limits': ['Includes Cargo normal and build edges, excludes dev edges; build-only packages can be a conservative superset.', 'Cargo metadata and crate notices do not by themselves establish embedded data provenance or all notice obligations.']}
(out / 'dependency-notices.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
