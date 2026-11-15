#!/usr/bin/env python3
"""Exercise the packaged CPU CLI using only synthetic data."""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

binary = pathlib.Path(sys.argv[1]).resolve()
legacy_dir = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None

def invoke(executable, args, text=None):
    env = os.environ.copy()
    env['PATH'] = str(pathlib.Path(env['SystemRoot']) / 'System32') if os.name == 'nt' else '/usr/bin:/bin'
    return subprocess.run([str(executable), *map(str, args)], input=text, text=True, capture_output=True, encoding='utf-8', cwd=root, env=env)

def run(args, text=None):
    result = invoke(binary, args, text)
    if result.returncode:
        raise RuntimeError(f'{args}: {result.returncode}\n{result.stderr}')
    return result.stdout

def legacy(command):
    name = 'nlp_' + command.replace('-', '_')
    return legacy_dir / (name + ('.exe' if os.name == 'nt' else ''))

def equivalent(command, args, text=None):
    current = invoke(binary, [command, *args], text)
    old = invoke(legacy(command), args, text)
    assert (current.returncode, current.stdout, current.stderr) == (old.returncode, old.stdout, old.stderr), command
    assert current.returncode == 0, current.stderr
    return current.stdout

def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in {'elapsed_ms', 'elapsed_secs', 'chapters_per_sec', 'sentences_per_sec', 'processed_sentences_per_sec'}}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value

def files(directory):
    result = {}
    for p in directory.rglob('*'):
        if not p.is_file():
            continue
        content = p.read_bytes()
        if p.suffix == '.json':
            content = stable(json.loads(content))
        result[str(p.relative_to(directory))] = content
    return result

def batch(command, args, dest):
    old = invoke(legacy(command), args)
    assert old.returncode == 0, old.stderr
    before = files(dest)
    # Keep the old results for comparison. The new command must create every
    # output itself; existing files must not hide a missing write.
    dest.rename(dest.with_name(dest.name + '.legacy'))
    current = invoke(binary, [command, *args])
    assert current.returncode == 0, current.stderr
    assert current.stdout == old.stdout
    def normalize_progress(value):
        return re.sub(r'elapsed=[0-9.]+s chapters/sec=[0-9.]+ sents/sec=[0-9.]+ ETA=[0-9.]+s', 'TIMING', value)
    assert normalize_progress(current.stderr) == normalize_progress(old.stderr)
    assert before == files(dest), command

with tempfile.TemporaryDirectory(prefix='nlp-release-smoke-') as tmp:
    root = pathlib.Path(tmp)
    source = root / 'input'
    source.mkdir()
    (source / 'chapter.txt').write_text('小明走到公園。\n', encoding='utf-8')
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps({'count': 1, 'chapters': [{'order': 1, 'title': '第一章', 'file': 'chapter.txt', 'status': 'ok'}]}, ensure_ascii=False), encoding='utf-8')
    for command in ('doc', 'novel-batch', 'source-stream', 'narrative-doc', 'narrative-batch', 'translate', 'translate-batch'):
        help_text = run([command, '--help'])
        assert f'Usage: nlp {command}' in help_text
        if legacy_dir:
            old_name = 'nlp_' + command.replace('-', '_')
            old_exe = legacy(command)
            old = subprocess.run([str(old_exe), '--help'], text=True, encoding='utf-8', capture_output=True, check=True)
            assert old.stdout.replace(old_name, 'nlp ' + command) == help_text
    assert legacy_dir, 'comparison requires the native legacy binaries'
    assert json.loads(equivalent('doc', [], '這是測試。'))
    assert json.loads(equivalent('narrative-doc', [], '這是測試。'))
    assert equivalent('translate', ['--target', 'zh-Hant', '--source-hint', 'ja', '--format', 'text'], 'これはテストです。').strip()
    assert equivalent('source-stream', ['--manifest', manifest, '--all']).strip()
    analysis, narrative, translated = root / 'analysis', root / 'narrative', root / 'translated'
    batch('novel-batch', ['--manifest', manifest, '--out-dir', analysis, '--backend', 'none', '--workers', '1', '--no-resume'], analysis)
    assert (analysis / 'chapters' / '0001.json').is_file()
    batch('narrative-batch', ['--analysis-dir', analysis, '--bundles-dir', source, '--out-dir', narrative, '--workers', '1', '--no-resume'], narrative)
    assert (narrative / 'chapters' / '0001.json').is_file()
    batch('translate-batch', ['--input-dir', source, '--output-dir', translated, '--target', 'zh-Hant', '--workers', '1', '--no-resume'], translated)
    assert (translated / 'text' / 'chapter.txt').is_file()
    missing = invoke(binary, ['source-stream', '--manifest', root / 'missing.json', '--all'])
    missing_old = invoke(legacy('source-stream'), ['--manifest', root / 'missing.json', '--all'])
    assert missing.returncode != 0 and (missing.returncode, missing.stdout, missing.stderr) == (missing_old.returncode, missing_old.stdout, missing_old.stderr)
print('all seven CPU commands passed synthetic smoke')
