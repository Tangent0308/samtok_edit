"""Resumable multi-GPU frozen SAMTok region cache for full training metadata."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .data import file_hash, load_images, row_hash, write_json
from .protocol import spans_in, validate_row
from .region_supervision import (GEOMETRY, SCHEMA, coverage_grid, local_umt,
                                 validate_supervision)


def metadata_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield validate_row(json.loads(line))


def identity_for(metadata, qwen, samtok, max_pixels, processor):
    from .model import resize_sources
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Unit_EditImageEmbedder

    min_pixels = QwenImage21Unit_EditImageEmbedder.get_processor_min_pixels(
        SimpleNamespace(processor=processor))
    model = Path(samtok)
    identity = {'schema': SCHEMA, 'geometry': GEOMETRY, 'max_pixels': max_pixels,
                'codec': file_hash(model / 'mask_tokenizer_256x2.pth'),
                'sam2': file_hash(model / 'sam2.1_hiera_large.pt'),
                'metadata_sha256': file_hash(metadata), 'skip_empty': True,
                'alignment': 'certified-full-frame', 'source_min_pixels': min_pixels}
    return identity


def worker(metadata, output, qwen, samtok, rank, shards, device, max_pixels):
    from .codec import SamtokCodec
    from .model import build_processor, resize_sources

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not 0 <= rank < shards:
        raise ValueError('Invalid region shard rank')
    metadata, output = Path(metadata).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    processor = build_processor(qwen, samtok)
    identity = identity_for(metadata, qwen, samtok, max_pixels, processor)
    digest = row_hash(identity)
    model = Path(samtok)
    codec = SamtokCodec(model / 'sam2.1_hiera_large.pt', model / 'mask_tokenizer_256x2.pth',
                        device=device)
    records, seen, image_hashes = {}, 0, {}
    counts = Counter()
    coverage_dir = output / 'coverage'
    coverage_dir.mkdir(exist_ok=True)

    def coverage_path(row):
        digest_key = json.dumps({'source': row['edit_image'], 'target': row['image'],
                                 'spans': spans_in(row['prompt']), 'max_pixels': max_pixels,
                                 'codec': identity['codec']}, sort_keys=True).encode()
        return coverage_dir / (hashlib.sha256(digest_key).hexdigest() + '.pt')

    def register(row, key, value, path=None):
        nonlocal seen
        counts['eligible' if value.get('eligible') else value['reason']] += 1
        if path is None:
            records[key] = {'eligible': False, 'reason': value['reason']}
        else:
            records[key] = {'file': path.name, 'sha256': file_hash(path),
                            'eligible': value['eligible'], 'reason': value['reason']}

    def save_new(row, key, masks, resized, height, width):
        path = output / f'{key}.pt'
        spans = spans_in(row['prompt'])
        if len(masks) != len(spans):
            raise ValueError('Decoded mask/span count mismatch')
        value = {'schema': SCHEMA, 'identity': digest, 'row_hash': key,
                 'eligible': True, 'reason': ''}
        source = torch.stack([coverage_grid(mask, resized.height, resized.width)
                              for mask in masks])
        target_grid = torch.stack([coverage_grid(mask, height, width)
                                   for mask in masks])
        if (source.flatten(1).amax(1) <= 0).any() or (target_grid.flatten(1).amax(1) <= 0).any():
            value.update(eligible=False, reason='empty_region')
        else:
            value.update(spans=spans, coverage_source=source,
                         coverage_target=target_grid)
        names = row['edit_image']
        names = [names] if isinstance(names, str) else names
        for name in [*names, row['image']]:
            if name not in image_hashes:
                image_hashes[name] = file_hash(name)
        value['images'] = {name: image_hashes[name] for name in [*names, row['image']]}
        coverage = coverage_path(row)
        if not coverage.exists():
            temporary_coverage = coverage.with_suffix('.tmp')
            torch.save({'spans': spans, 'coverage_source': value.get('coverage_source'),
                        'coverage_target': value.get('coverage_target'),
                        'images': value.get('images', {})}, temporary_coverage)
            try:
                temporary_coverage.replace(coverage)
            except FileNotFoundError:
                # A ref and noref row can reach the same content key on two
                # ranks. The winner's identical coverage is authoritative.
                if not coverage.exists():
                    raise
        validate_supervision(value, row=row)
        temporary = output / f'{key}.{rank}.tmp'
        torch.save(value, temporary)
        temporary.replace(path)
        register(row, key, value, path)

    def from_coverage(row, key, cached):
        path = output / f'{key}.pt'
        value = {'schema': SCHEMA, 'identity': digest, 'row_hash': key,
                 'eligible': True, 'reason': '', 'spans': cached['spans'],
                 'coverage_source': cached['coverage_source'],
                 'coverage_target': cached['coverage_target'], 'images': cached['images']}
        validate_supervision(value, row=row)
        temporary = path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(path)
        register(row, key, value, path)

    def flush(batch):
        if not batch:
            return
        single, multi = [], []
        for row, key in batch:
            sources, target, height, width = load_images(row, '/', max_pixels)
            if len(sources) != 1:
                raise ValueError('Region supervision requires one source image')
            resized = resize_sources(SimpleNamespace(processor=processor), sources,
                                     height, width)[0]
            entry = (row, key, sources[0], resized, height, width)
            (single if len(spans_in(row['prompt'])) == 1 else multi).append(entry)
        for offset in range(0, len(single), 16):
            group = single[offset:offset + 16]
            decoded = codec.decode_single_batch([(x[2], x[0]['prompt']) for x in group])
            for entry, masks in zip(group, decoded):
                save_new(entry[0], entry[1], [masks], entry[3], entry[4], entry[5])
        for row, key, source, resized, height, width in multi:
            save_new(row, key, codec.decode_strict(source, row['prompt']), resized, height, width)

    pending = []
    for row in metadata_rows(metadata):
        key = row_hash(row)
        if int(key[:8], 16) % shards != rank or key in records:
            continue
        seen += 1
        path = output / f'{key}.pt'
        if not local_umt(row):
            value = {'schema': SCHEMA, 'identity': digest, 'row_hash': key,
                     'eligible': False, 'reason': 'task'}
            validate_supervision(value, row=row)
            register(row, key, value)
        elif path.exists():
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value.get('identity') != digest:
                raise ValueError(f'Stale region cache file: {path}')
            validate_supervision(value, row=row)
            if value.get('eligible') and not coverage_path(row).exists():
                coverage = coverage_path(row)
                temporary = coverage.with_suffix('.tmp')
                torch.save({'spans': value['spans'], 'coverage_source': value['coverage_source'],
                            'coverage_target': value['coverage_target'], 'images': value.get('images', {})},
                           temporary)
                try:
                    temporary.replace(coverage)
                except FileNotFoundError:
                    if not coverage.exists():
                        raise
            register(row, key, value, path)
        else:
            cached = coverage_path(row)
            if cached.exists():
                from_coverage(row, key, torch.load(cached, map_location='cpu', weights_only=True))
            else:
                pending.append((row, key))
                if len(pending) >= 16:
                    flush(pending)
                    pending = []
        if seen % 100 == 0:
            if pending:
                flush(pending)
                pending = []
            write_json(output / f'progress-{rank:02d}.json',
                       {'rank': rank, 'rows': seen, 'counts': dict(counts)})
            print(json.dumps({'rank': rank, 'rows': seen, 'counts': dict(counts)}), flush=True)
    flush(pending)
    write_json(output / f'shard-{rank:02d}.json',
               {'rank': rank, 'shards': shards, 'identity': identity,
                'identity_hash': digest, 'counts': dict(counts), 'rows': records})
    print(json.dumps({'rank': rank, 'rows': seen, 'complete': True,
                      'counts': dict(counts)}), flush=True)


def merge(metadata, output, shards):
    metadata, output = Path(metadata).resolve(), Path(output).resolve()
    all_rows, identities, counts = {}, [], Counter()
    for rank in range(shards):
        value = json.loads((output / f'shard-{rank:02d}.json').read_text())
        if value['rank'] != rank or value['shards'] != shards:
            raise ValueError('Wrong region shard identity')
        identities.append(value['identity'])
        for key, record in value['rows'].items():
            if key in all_rows:
                raise ValueError('Duplicate region row across shards')
            if 'file' in record and not (output / record['file']).is_file():
                raise ValueError('Missing region tensor file')
            if 'file' not in record and record != {'eligible': False, 'reason': 'task'}:
                raise ValueError('Only task-ineligible rows may omit a tensor file')
            all_rows[key] = record
        counts.update(value['counts'])
    if any(x != identities[0] for x in identities):
        raise ValueError('Region shard identities differ')
    if identities[0]['metadata_sha256'] != file_hash(metadata):
        raise ValueError('Region metadata changed')
    expected = {row_hash(row) for row in metadata_rows(metadata)}
    if set(all_rows) != expected:
        raise ValueError(f'Region cache covers {len(all_rows)} of {len(expected)} unique metadata rows')
    digest = row_hash(identities[0])
    write_json(output / 'manifest.json', {'identity': identities[0],
                                          'identity_hash': digest, 'rows': all_rows})
    report = {'rows': len(all_rows), 'counts': dict(counts), 'identity_hash': digest,
              'metadata_sha256': identities[0]['metadata_sha256'],
              'manifest_sha256': file_hash(output / 'manifest.json')}
    write_json(output / 'report.json', report)
    metadata_report = output.parent / 'metadata_report.json'
    if metadata_report.is_file():
        prepared = json.loads(metadata_report.read_text())
        if prepared['stage1_sha256'] != report['metadata_sha256']:
            raise ValueError('Region cache is not bound to stage1 metadata')
        prepared.update(training_ready=True, region_cache_ready=True,
                        region_manifest_sha256=report['manifest_sha256'],
                        region_counts=report['counts'])
        write_json(metadata_report, prepared)
    print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    worker_parser = sub.add_parser('worker')
    worker_parser.add_argument('--metadata', required=True)
    worker_parser.add_argument('--output', required=True)
    worker_parser.add_argument('--qwen', required=True)
    worker_parser.add_argument('--samtok', required=True)
    worker_parser.add_argument('--rank', type=int, required=True)
    worker_parser.add_argument('--shards', type=int, required=True)
    worker_parser.add_argument('--device', default='cuda:0')
    worker_parser.add_argument('--max-pixels', type=int, default=1048576)
    merge_parser = sub.add_parser('merge')
    merge_parser.add_argument('--metadata', required=True)
    merge_parser.add_argument('--output', required=True)
    merge_parser.add_argument('--shards', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'worker':
        worker(args.metadata, args.output, args.qwen, args.samtok,
               args.rank, args.shards, args.device, args.max_pixels)
    else:
        merge(args.metadata, args.output, args.shards)


if __name__ == '__main__':
    main()
