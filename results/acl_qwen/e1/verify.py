"""Verify this portable E1 bundle with Python 3's standard library only."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    root = Path(__file__).resolve().parent
    for line in (root / 'SHA256SUMS').read_text(encoding='ascii').splitlines():
        digest, name = line.split('  ', 1)
        require(Path(name).name == name, 'Unsafe checksum filename')
        require(sha((root / name).read_bytes()) == digest, f'Checksum mismatch: {name}')
    rows = read_rows(root / 'reviews.jsonl.gz')
    papers = read_rows(root / 'manifest.jsonl.gz')
    inputs = read_rows(root / 'input_versions.jsonl.gz')
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    configs = json.loads((root / 'run_configs.json').read_text(encoding='utf-8'))
    input_hash = {(r['arxiv_id'], r['v']): r['text_sha256'] for r in inputs}
    prompt_sha = sha((root / 'review_prompt.md').read_bytes())
    manifest_sha = sha(gzip.decompress((root / 'manifest.jsonl.gz').read_bytes()))
    require(manifest_sha == provenance['source_manifest_sha256'], 'Manifest hash mismatch')
    expected = {(p['arxiv_id'], v['v'], r) for p in papers for v in p['versions'] for r in range(3)}
    actual = {(r['arxiv_id'], r['v'], r['rep']) for r in rows}
    unavailable = {(a, v, r) for a, v in [('2402.14228', 3), ('2502.12509', 4)] for r in range(3)}
    require(len(papers) == 1379 and len(expected) == 10887, 'Original population changed')
    require(len(rows) == len(actual) == 10881, 'Incorrect count or duplicate observations')
    require(expected - actual == unavailable and not actual - expected, 'Unexpected missing/extra observations')
    require(len(inputs) == len(input_hash) == 3627, 'Incorrect input version count')
    repetitions = defaultdict(set)
    by_raw = {}
    for row in rows:
        key = (row['arxiv_id'], row['v'])
        repetitions[key].add(row['rep'])
        require(row['parsed_ok'] and not row['error'] and not row['truncated'] and not row['mock'], f'Invalid row: {key}')
        require(row['api_model'] == row['response_model'] == 'qwen-e1', 'Model alias mismatch')
        require(row['finish_reason'] == 'stop', 'Incomplete generation')
        require(row['temperature'] == 1 and row['chat_template_kwargs'] == {'enable_thinking': False}, 'Inference settings changed')
        require(row['text_sha'] == input_hash[key] and row['prompt_sha'] == prompt_sha, f'Input identity mismatch: {key}')
        config = configs[row['model']]['config']
        require(config['prompt_sha256'] == prompt_sha and config['manifest_sha256'] == manifest_sha, 'Config input identity mismatch')
        require(config['runtime']['model_sha256'] == '0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5', 'Model hash mismatch')
        require(row['raw_path'] not in by_raw, 'Repeated raw path')
        by_raw[row['raw_path']] = row
    require(all(reps == {0, 1, 2} for reps in repetitions.values()), 'Incomplete repetitions')
    seen = set()
    warnings = []
    ranges = {'rating': {1, 3, 5, 6, 8, 10}, 'soundness': {1, 2, 3, 4},
              'presentation': {1, 2, 3, 4}, 'contribution': {1, 2, 3, 4}, 'confidence': {1, 2, 3, 4, 5}}
    with tarfile.open(root / 'raw_responses.tar.gz', 'r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts, 'Unsafe archive member')
            require(member.name in by_raw and member.name not in seen, 'Unexpected/duplicate raw member')
            seen.add(member.name)
            row = by_raw[member.name]
            raw_bytes = archive.extractfile(member).read()
            require(sha(raw_bytes) == row['raw_sha256'], 'Raw bytes hash mismatch')
            raw = json.loads(raw_bytes)
            choice = raw['choices'][0]
            require(raw['model'] == 'qwen-e1' and choice['finish_reason'] == 'stop', 'Raw model/finish mismatch')
            require(not choice['message'].get('reasoning_content'), 'Unexpected thinking output')
            content = choice['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            review = json.loads(content)
            require(review == row['review'], 'Parsed narrative differs from original')
            for field, valid in ranges.items():
                require(type(review[field]) is int and review[field] in valid and review[field] == row[field], f'Invalid/mismatching score: {field}')
            require(type(review['accept']) is bool and review['accept'] == row['accept'], 'Invalid accept vote')
            if not all(f in review for f in ('summary', 'strengths', 'weaknesses', 'questions')):
                warnings.append((row['arxiv_id'], row['v'], row['rep']))
    require(seen == set(by_raw), 'Missing original responses')
    require(warnings == [('2502.11364', 2, 1)], 'Narrative warning set changed')
    counts = dict(Counter(r['model'] for r in rows))
    require(counts == provenance['export']['counts_by_label'], 'Runtime profile count mismatch')
    print(json.dumps({'status': 'verified', 'valid_reviews': len(rows), 'available_versions': len(inputs),
                      'papers_with_results': len({r['arxiv_id'] for r in rows}),
                      'complete_paper_chains': len(papers) - len({r[0] for r in unavailable}),
                      'missing_withdrawn_reviews': len(unavailable), 'exact_raw_responses': len(seen),
                      'known_narrative_warnings': len(warnings)}, indent=2))


if __name__ == '__main__':
    main()
