"""Build the portable E1 result bundle from frozen local logs (stdlib only).

Run from a checkout containing data/processed/acl_qwen_recovery and the
original raw responses/texts. Does not modify any experiment inputs or logs.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/processed/acl_qwen_recovery'
SOURCE_SHA = 'ce22eafab7b7ca053f7c3194b99ff8a40096e2851516059176c53ba7a0a393ec'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def compressed(path, data):
    with path.open('wb') as out:
        with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=6) as gz:
            gz.write(data)


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        normalized = value.replace('\\', '/')
        prefix = ROOT.as_posix() + '/'
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/acl_qwen/e1')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not all((out / f).is_file() for f in ('README.md', 'verify.py', '.gitattributes')):
        raise RuntimeError('Output must contain the versioned README.md, verify.py and .gitattributes')

    original = (SOURCE / 'reviews_combined.jsonl').read_bytes()
    if sha(original) != SOURCE_SHA:
        raise RuntimeError('Frozen source combined log changed')
    rows = [json.loads(line) for line in original.splitlines()]
    audit = read_json(SOURCE / 'final_audit.json')
    configs = {}
    for label, evidence in audit['runtime_configs'].items():
        path = ROOT / 'data/processed/e1_qwen/acl' / label / 'config.json'
        content = path.read_bytes()
        if sha(content) != evidence['sha256']:
            raise RuntimeError(f'Runtime config changed: {label}')
        configs[label] = dict(original_sha256=sha(content), config=portable(json.loads(content)))

    inputs = {}
    exported = []
    names = set()
    with (out / 'raw_responses.tar.gz').open('wb') as target:
        with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for row in rows:
                    label = row['model']
                    # Resolve from the log's known layout, never follow arbitrary paths.
                    name = row['raw_path'].replace('\\', '/').rsplit('/', 1)[-1]
                    raw_path = ROOT / 'data/processed/e1_qwen/acl' / label / 'raw' / label / name
                    raw_bytes = raw_path.read_bytes()
                    if sha(raw_bytes) != row['raw_sha256']:
                        raise RuntimeError(f'Raw response changed: {name}')
                    member = f'raw/{label}/{name}'
                    if member in names:
                        raise RuntimeError(f'Duplicate raw response: {member}')
                    names.add(member)
                    info = tarfile.TarInfo(member)
                    info.size = len(raw_bytes)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(raw_bytes))
                    content = json.loads(raw_bytes)['choices'][0]['message']['content'].strip()
                    if content.startswith('```'):
                        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                    review = json.loads(content)
                    item = portable(row)
                    item['raw_path'] = member
                    item['review'] = review
                    exported.append(item)
                    key = (row['arxiv_id'], row['v'])
                    if key not in inputs:
                        text_path = ROOT / 'data/processed/acl_qwen_full/texts' / key[0] / f'v{key[1]}.md'
                        text_bytes = text_path.read_bytes()
                        inputs[key] = dict(arxiv_id=key[0], v=key[1], text_sha256=sha(text_bytes),
                                           word_count=len(text_bytes.decode('utf-8').split()),
                                           input_provenance='regenerated')
                    if inputs[key]['text_sha256'] != row['text_sha']:
                        raise RuntimeError(f'Input hash changed: {key}')

    compressed(out / 'reviews.jsonl.gz', jsonl(exported))
    compressed(out / 'input_versions.jsonl.gz', jsonl(inputs[k] for k in sorted(inputs)))
    manifest = (ROOT / 'data/processed/acl_manifest.jsonl').read_bytes()
    compressed(out / 'manifest.jsonl.gz', manifest)
    shutil.copyfile(ROOT / 'src/review/prompts/review_v1.md', out / 'review_prompt.md')
    for name in ('analysis.json', 'per_paper.jsonl', 'overview.png'):
        shutil.copyfile(ROOT / 'results/acl_qwen' / name, out / name)
    write_json(out / 'run_configs.json', configs)
    evidence_names = ('final_audit.json', 'independent_final_audit.json', 'input_audit.json',
                      'unavailable_versions.json', 'baseline_preservation_audit.json')
    write_json(out / 'provenance.json', {
        'source_combined_sha256': SOURCE_SHA,
        'source_manifest_sha256': sha(manifest),
        'original_evidence_sha256': {n: sha((SOURCE / n).read_bytes()) for n in evidence_names},
        'evidence': {n: portable(read_json(SOURCE / n)) for n in evidence_names},
        'export': {
            'schema_version': 1,
            'valid_reviews': len(exported),
            'input_versions': len(inputs),
            'counts_by_label': dict(Counter(r['model'] for r in exported)),
            'transformations': [
                'All original observation fields retained; absolute repository paths made relative.',
                'raw_path points to the exact-byte response in raw_responses.tar.gz.',
                'Added review: JSON parsed from raw content, removing only enclosing Markdown fences.',
                'Metadata evidence paths are repository-relative provenance, not bundled source files.',
                'No scores changed; narrative spelling warning retained; paper texts and model excluded.'
            ],
        },
    })
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in files),
                                   encoding='ascii', newline='\n')
    print(json.dumps({'output': str(out), 'reviews': len(exported),
                      'files': {p.name: p.stat().st_size for p in files}}, indent=2))


if __name__ == '__main__':
    main()
