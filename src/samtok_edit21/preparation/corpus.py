"""Resumable four-source training metadata builder using the released SAMTok codec.

The staged dataset mask is authoritative. Single-operation records use its
aggregate PNG; multi-operation records use only the selected existing instance
RLEs, one SAMTok span per instance. No masks are predicted or synthesized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from samtok_edit21.data.io import file_hash, row_hash, write_json
from samtok_edit21.preparation.converters import convert_record
from samtok_edit21.data.protocol import (
    EDIT_TYPES, grouped_units, parse_cot, render_units, validate_row,
)


COARSE_PLAIN_TYPES = {
    'compositional_editing': 'composite',
    'symbolic_reasoning': 'action',
    'scientific_reasoning': 'action',
    'perceptual_reasoning': 'action',
    'social_reasoning': 'action',
    'count_change': 'action',
}


def plain_type(source):
    value = source['provisional_type'] or COARSE_PLAIN_TYPES.get(source['native_type'])
    if value not in EDIT_TYPES:
        raise ValueError(f"No plain edit type for {source['id']}: {source['native_type']}")
    return value


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def split(source_root, semantic_run, output, workers, sample_per_dataset=0):
    """Bind immutable semantic IDs to all quality-accepted image/mask rows."""
    source_root, semantic_run, output = map(lambda p: Path(p).resolve(),
                                            (source_root, semantic_run, output))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'Input split must use a fresh directory: {output}')
    inventory = json.loads((source_root / 'source_inventory.json').read_text())
    report = json.loads((semantic_run / 'conversion_report.json').read_text())
    source_path = source_root / 'sources.jsonl'
    annotation_path = semantic_run / 'annotations.jsonl'
    failed_path = semantic_run / 'failed.jsonl'
    if file_hash(source_path) != inventory['source_manifest_sha256']:
        raise ValueError('Source manifest hash differs from inventory')
    if file_hash(annotation_path) != report['annotations_sha256']:
        raise ValueError('Semantic annotation hash differs from report')
    if not (semantic_run / 'SUCCESS.json').is_file():
        raise ValueError('Semantic run has no SUCCESS marker')
    annotations = {}
    for item in read_jsonl(annotation_path):
        if item['id'] in annotations or item['status'] != 'accepted':
            raise ValueError('Duplicate or nonaccepted annotation')
        annotations[item['id']] = item
    failed = set()
    for item in read_jsonl(failed_path):
        uid = item['source']['id']
        if uid in failed or item['result']['status'] != 'failed':
            raise ValueError('Duplicate or nonfailed failed record')
        failed.add(uid)
    if len(annotations) != report['accepted_count'] or len(failed) != report['failed_count']:
        raise ValueError('Semantic report count mismatch')
    if annotations.keys() & failed:
        raise ValueError('ID is both accepted and failed')
    streams = [(output / f'input-{rank:02d}.jsonl').open('w') for rank in range(workers)]
    counts, kept = Counter(), Counter()
    seen = set()
    try:
        for index, source in enumerate(read_jsonl(source_path)):
            uid, dataset = source['id'], source['dataset']
            if uid in seen or uid not in annotations and uid not in failed:
                raise ValueError(f'Missing/duplicate semantic ID {uid}')
            seen.add(uid)
            counts[(dataset, 'source')] += 1
            annotation = annotations.get(uid)
            if annotation is not None:
                counts[(dataset, annotation['conversion_method'])] += 1
            else:
                counts[(dataset, 'plain_only')] += 1
            if sample_per_dataset and kept[dataset] >= sample_per_dataset:
                continue
            kept[dataset] += 1
            if source.get('quality_accepted') is not True:
                raise ValueError(f'Source failed quality filtering: {uid}')
            if annotation is not None and annotation['id'] != uid:
                raise ValueError(f'Wrong semantic annotation for {uid}')
            record = {'index': index, 'source': source,
                      'annotation': annotation['annotation'] if annotation else None,
                      'conversion_method': annotation['conversion_method'] if annotation else 'plain_only'}
            streams[index % workers].write(json.dumps(record, ensure_ascii=False) + '\n')
    finally:
        for stream in streams:
            stream.close()
    if len(seen) != inventory['total_accepted'] or seen != annotations.keys() | failed:
        raise ValueError('Source/semantic ID coverage mismatch')
    if {ds: counts[(ds, 'source')] for ds in inventory['accepted']} != inventory['accepted']:
        raise ValueError('Source dataset counts mismatch')
    identity = {'source_root': str(source_root), 'semantic_run': str(semantic_run),
                'source_sha256': inventory['source_manifest_sha256'],
                'annotations_sha256': report['annotations_sha256'],
                'failed_sha256': file_hash(failed_path), 'workers': workers,
                'sample_per_dataset': sample_per_dataset,
                'selected': dict(kept), 'full_counts': {f'{ds}:{kind}': n for (ds, kind), n in counts.items()},
                'input_sha256': {f'input-{rank:02d}': file_hash(output / f'input-{rank:02d}.jsonl')
                                 for rank in range(workers)}}
    write_json(output / 'inputs.json', identity)
    print(json.dumps({'selected': dict(kept), 'full_counts': identity['full_counts']}), flush=True)


def decode_instance(instance):
    rle = {'size': instance['rle_size'], 'counts': instance['rle_counts'].encode('ascii')}
    result = coco_mask.decode(rle)
    if result.ndim != 2:
        raise ValueError('Expected one 2D instance mask')
    return result


def masks_for_record(source, units):
    if len(units) == 1 and units[0]['mask_ids'] == ['union']:
        if source['dataset_mask']:
            with Image.open(source['dataset_mask']) as image:
                return [[np.asarray(image.convert('L')) > 0]]
        if len(source['instances']) != 1:
            raise ValueError('Derived aggregate requires one stored instance')
        return [[decode_instance(source['instances'][0])]]
    by_id = {item['instance_id']: item for item in source['instances']}
    if len(by_id) != len(source['instances']):
        raise ValueError('Duplicate source instance ID')
    selected = [uid for unit in units for uid in unit['mask_ids']]
    if len(selected) != len(set(selected)) or set(selected) != set(by_id):
        raise ValueError('Composite mask IDs must partition the supplied instances')
    return [[decode_instance(by_id[uid]) for uid in unit['mask_ids']] for unit in units]


def encode_chunk(records, codec, batch_size):
    """Encode existing dataset masks in batches; validate all derived train rows."""
    pairs, slices = [], []
    for item in records:
        source, annotation = item['source'], item['annotation']
        if annotation is None:
            slices.append(None)
            continue
        with Image.open(source['edit_image']) as image:
            image = image.copy()
        masks = masks_for_record(source, annotation['units'])
        start = len(pairs)
        for group in masks:
            if len(group) > 1:
                _, order = codec._ordered_masks(group)
                group = [group[index] for index in order]
            for mask in group:
                pairs.append((image, mask))
        slices.append((start, [len(group) for group in masks]))
    codes = []
    for offset in range(0, len(pairs), batch_size):
        codes.extend(codec.encode_single_batch(pairs[offset:offset + batch_size]))
    result = []
    for item, assignment in zip(records, slices):
        source, annotation = item['source'], item['annotation']
        if annotation is None:
            rows = [validate_row({'sample_type': 'edit', 'edit_type': plain_type(source),
                                  'edit_image': source['edit_image'], 'image': source['image'],
                                  'prompt': source['instruction']})]
        else:
            cursor, lengths = assignment
            units = []
            for unit, length in zip(annotation['units'], lengths):
                units.append({'edit_type': unit['edit_type'], 'ref_phrase': unit['ref_phrase'],
                              'mask_codes': codes[cursor:cursor + length],
                              'anchor_phrase': unit.get('anchor_phrase')})
                cursor += length
            rows, errors = convert_record({'instruction': source['instruction'],
                                           'edit_image': source['edit_image'], 'image': source['image'],
                                           'units': units,
                                           'noref_instruction': annotation['noref_instruction']})
            kinds = [(row['sample_type'], row.get('instr_variant')) for row in rows]
            expected = [('edit_ntp', None), ('edit', None),
                        ('edit_umt', 'ref'), ('edit_umt', 'noref')]
            # convert_record keeps one noref row when both UMT prompts are
            # identical (typically a global edit). Verify the omitted ref
            # prompt rather than rejecting valid deduplication or accepting
            # a genuinely missing ref row.
            if not errors and kinds == [expected[0], expected[1], expected[3]]:
                reference = render_units(source['instruction'], grouped_units(
                    source['instruction'], parse_cot(rows[0]['mt_cot'])))
                if reference == rows[-1]['prompt']:
                    expected.pop(2)
            if errors or kinds != expected:
                raise ValueError(f"Incomplete derived rows for {source['id']}: {errors}; {kinds}")
        result.append({'index': item['index'], 'id': source['id'], 'dataset': source['dataset'],
                       'conversion_method': item['conversion_method'], 'rows': rows})
    return result


def encode_worker(output, rank, codec_root, device, chunk_size=64, batch_size=8):
    import torch
    from samtok_edit21.models.codec import SamtokCodec

    # Eight independent codec processes otherwise inherit PyTorch's 96-core
    # default, spawning hundreds of threads each for small CPU mask operations.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(output).resolve()
    identity = json.loads((output / 'inputs.json').read_text())
    if not 0 <= rank < identity['workers']:
        raise ValueError('Invalid worker rank')
    input_path = output / f'input-{rank:02d}.jsonl'
    if file_hash(input_path) != identity['input_sha256'][f'input-{rank:02d}']:
        raise ValueError('Worker input changed')
    target = output / 'encoded' / f'worker-{rank:02d}'
    target.mkdir(parents=True, exist_ok=True)
    model = Path(codec_root)
    codec = SamtokCodec(model / 'sam2.1_hiera_large.pt', model / 'mask_tokenizer_256x2.pth',
                        device=device)
    counts = Counter()
    with input_path.open() as stream:
        chunk = []
        for line in stream:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                _write_chunk(target, counts['chunks'], chunk, codec, batch_size)
                counts.update({'chunks': 1, 'sources': len(chunk)})
                if counts['chunks'] % 10 == 0:
                    print(json.dumps({'rank': rank, **counts}), flush=True)
                chunk = []
        if chunk:
            _write_chunk(target, counts['chunks'], chunk, codec, batch_size)
            counts.update({'chunks': 1, 'sources': len(chunk)})
    write_json(target / 'complete.json', {'rank': rank, **counts,
               'input_sha256': identity['input_sha256'][f'input-{rank:02d}'],
               'codec_sha256': file_hash(model / 'mask_tokenizer_256x2.pth')})
    print(json.dumps({'rank': rank, **counts, 'complete': True}), flush=True)


def _write_chunk(target, number, records, codec, batch_size):
    path = target / f'chunk-{number:05d}.jsonl'
    receipt = target / f'chunk-{number:05d}.receipt.json'
    input_hash = row_hash([(item['index'], item['source']['id']) for item in records])
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['input_hash'] == input_hash and saved['sha256'] == file_hash(path):
            return
        raise ValueError(f'Corrupt/stale chunk receipt: {receipt}')
    converted = encode_chunk(records, codec, batch_size)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for item in converted:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    temporary.replace(path)
    write_json(receipt, {'input_hash': input_hash, 'sha256': file_hash(path),
                         'sources': len(converted), 'rows': sum(len(x['rows']) for x in converted)})


def worker_rows(output, rank):
    target = output / 'encoded' / f'worker-{rank:02d}'
    complete = json.loads((target / 'complete.json').read_text())
    for number in range(complete['chunks']):
        path = target / f'chunk-{number:05d}.jsonl'
        receipt = json.loads((target / f'chunk-{number:05d}.receipt.json').read_text())
        if file_hash(path) != receipt['sha256']:
            raise ValueError(f'Encoded chunk checksum mismatch: {path}')
        yield from read_jsonl(path)


def merge(output):
    output = Path(output).resolve()
    identity = json.loads((output / 'inputs.json').read_text())
    codec_hashes = set()
    for rank in range(identity['workers']):
        complete = json.loads((output / 'encoded' / f'worker-{rank:02d}' / 'complete.json').read_text())
        if complete['input_sha256'] != identity['input_sha256'][f'input-{rank:02d}']:
            raise ValueError('Encoded worker input identity mismatch')
        codec_hashes.add(complete['codec_sha256'])
    if len(codec_hashes) != 1:
        raise ValueError('Workers used different SAMTok codecs')
    stage1 = output / 'stage1.jsonl'
    stage2 = output / 'stage2.jsonl'
    provenance = output / 'provenance.jsonl'
    counts = Counter()
    sources = 0
    seen_indices = set()
    streams = [p.with_suffix('.tmp').open('w') for p in (stage1, stage2, provenance)]
    try:
        rows = heapq.merge(*(worker_rows(output, rank) for rank in range(identity['workers'])),
                           key=lambda item: item['index'])
        for item in rows:
            if item['index'] in seen_indices:
                raise ValueError('Duplicate source index in encoded data')
            seen_indices.add(item['index'])
            sources += 1
            for row in item['rows']:
                validate_row(row)
                kind = row['sample_type'] + (':' + row['instr_variant']
                                             if row['sample_type'] == 'edit_umt' else '')
                counts[(item['dataset'], kind)] += 1
                source_ref = {'id': item['id'], 'dataset': item['dataset'],
                              'conversion_method': item['conversion_method'],
                              'sample_type': kind, 'edit_type': row['edit_type'],
                              'row_hash': row_hash(row)}
                streams[0].write(json.dumps(row, ensure_ascii=False) + '\n')
                streams[2].write(json.dumps(source_ref, ensure_ascii=False) + '\n')
                if row['sample_type'] != 'edit_ntp':
                    streams[1].write(json.dumps(row, ensure_ascii=False) + '\n')
    finally:
        for stream in streams:
            stream.close()
    expected = sum(identity['selected'].values())
    if sources != expected:
        raise ValueError(f'Merged {sources} sources; expected {expected}')
    for path in (stage1, stage2, provenance):
        path.with_suffix('.tmp').replace(path)
    report = {'sources': sources, 'selected_by_dataset': identity['selected'],
              'source_kind_counts': {f'{ds}:{kind}': n for (ds, kind), n in counts.items()},
              'stage1_rows': sum(counts.values()),
              'stage2_rows': sum(n for (_, kind), n in counts.items() if kind != 'edit_ntp'),
              'stage1_sha256': file_hash(stage1), 'stage2_sha256': file_hash(stage2),
              'provenance_sha256': file_hash(provenance),
              'codec_sha256': next(iter(codec_hashes)),
              'training_ready': False, 'region_cache_ready': False,
              'semantic_run': identity['semantic_run'], 'source_root': identity['source_root']}
    write_json(output / 'metadata_report.json', report)
    print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    split_parser = sub.add_parser('split')
    split_parser.add_argument('--source-root', required=True)
    split_parser.add_argument('--semantic-run', required=True)
    split_parser.add_argument('--output', required=True)
    split_parser.add_argument('--workers', type=int, default=8)
    split_parser.add_argument('--sample-per-dataset', type=int, default=0)
    worker = sub.add_parser('encode-worker')
    worker.add_argument('--output', required=True)
    worker.add_argument('--rank', type=int, required=True)
    worker.add_argument('--codec-root', required=True)
    worker.add_argument('--device', default='cuda:0')
    worker.add_argument('--chunk-size', type=int, default=64)
    worker.add_argument('--batch-size', type=int, default=8)
    merger = sub.add_parser('merge')
    merger.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'split':
        split(args.source_root, args.semantic_run, args.output, args.workers, args.sample_per_dataset)
    elif args.command == 'encode-worker':
        encode_worker(args.output, args.rank, args.codec_root, args.device,
                      args.chunk_size, args.batch_size)
    else:
        merge(args.output)


if __name__ == '__main__':
    main()
