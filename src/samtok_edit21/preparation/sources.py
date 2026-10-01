"""Materialize final-pass editing pairs without re-segmenting dataset masks.

This is the lossless first stage of full-data preparation. A source manifest is
NOT training metadata: semantic unit annotation and codec encoding follow it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from samtok_edit21.data.io import file_hash, write_json
from samtok_edit21.preparation.converters import native_edit_type

BASE = Path('/mnt/bn/strategy-mllm-train/user/tanyue')
DATASETS = {
    'refedit': BASE / 'datasets/RefEdit-mask-prefiltered-qwen38-self-contained/data',
    'crispedit': BASE / 'CrispEdit-labeling/final_dataset_39k/shards',
    'scaleedit': BASE / 'scaleedit_25k/shards',
    'derived': BASE / 'datasets/SAMTok_Derived_Edit_Labeling/combined',
}


def accepted(dataset, row):
    if dataset == 'refedit':
        return (row.get('prefilter_verdict') == 'PASS' and
                row.get('grounding_status') == 'OK' and row.get('qc_flag') == 'OK')
    if dataset == 'crispedit':
        return (row.get('quality__prefilter_verdict') == 'PASS' and
                row.get('scene__scene_pass') is True and row.get('mask__qc_flag') == 'OK')
    if dataset == 'scaleedit':
        return (row.get('quality__verdict') == 'PASS' and row.get('quality__keep') is True
                and row.get('scene__verdict') == 'PASS' and row.get('scene__keep') is True
                and row.get('mask__qc_flag') == 'OK')
    if dataset == 'derived':
        return row.get('planning_status') == 'accepted' and row.get('audit_status') == 'pass'
    raise ValueError(dataset)


def image_asset(value, output):
    """Decode every image; retain original bytes, including target alpha."""
    if isinstance(value, dict):
        value = value.get('bytes') or value.get('path')
    if isinstance(value, str):
        path = Path(value).resolve()
        with Image.open(path) as im:
            im.load()
            size, mode = list(im.size), im.mode
        return str(path), size, mode, file_hash(path)
    if not isinstance(value, bytes) or not value:
        raise ValueError('Image has neither readable path nor bytes')
    with Image.open(io.BytesIO(value)) as im:
        im.load()
        size, mode = list(im.size), im.mode
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp')
    tmp.write_bytes(value)
    tmp.replace(output)
    return str(output), size, mode, hashlib.sha256(value).hexdigest()


def parse_json(value):
    return json.loads(value) if isinstance(value, str) else value or {}


def source_id(dataset, locator):
    return dataset + '-' + hashlib.sha256(json.dumps(locator, sort_keys=True).encode()).hexdigest()[:24]


def semantic_record(dataset, row, locator):
    """Pure-text input; no image files or RLE needed for semantic conversion."""
    if dataset == 'derived':
        instruction, native = row['editing_instruction'], row['task_type']
        instances = [{'instance_id': 'selected_region',
                      'ref': row.get('region_contract', {}).get('segmentation_target')}]
        grounding = {'region_contract': row.get('region_contract')}
    else:
        instruction = row.get('final_instruction') or row['instruction']
        native = row.get('final_task', row.get('type'))
        prefix = '' if dataset == 'refedit' else 'mask__'
        instances = [{k: x.get(k) for k in ('instance_id', 'ref', 'grounding_image')}
                     for x in row[prefix + 'instance_masks']]
        full = parse_json(row.get(prefix + 'ground_json'))
        changes = full.get('observation', {}).get('parsed', {}).get('changes', [])
        grounding = {'observation': {'parsed': {'changes': [
            {k: x.get(k) for k in ('edit_id', 'source_ref', 'target_ref', 'change')} for x in changes]}}}
    return {'id': source_id(dataset, locator), 'dataset': dataset, 'locator': locator,
            'instruction': instruction, 'native_type': native,
            'provisional_type': native if dataset == 'derived' else native_edit_type(row),
            'instances': instances, 'grounding': grounding,
            'edit_image': 'annotation-only/source', 'image': 'annotation-only/target'}


def text_shard(job):
    dataset, path = job
    if dataset == 'derived':
        rows = (json.loads(line) for line in path.open())
    else:
        pf = pq.ParquetFile(path)
        names = {'sample_id', 'instruction', 'final_instruction', 'type', 'final_task',
                 'instance_masks', 'mask__instance_masks', 'ground_json', 'mask__ground_json',
                 'prefilter_verdict', 'grounding_status', 'qc_flag', 'quality__prefilter_verdict',
                 'scene__scene_pass', 'mask__qc_flag', 'quality__verdict', 'quality__keep',
                 'scene__verdict', 'scene__keep'}
        cols = [n for n in pf.schema_arrow.names if n in names]
        rows = (r for b in pf.iter_batches(columns=cols, batch_size=64) for r in b.to_pylist())
    result, total = [], 0
    for index, row in enumerate(rows):
        total += 1
        if accepted(dataset, row):
            locator = {'file': str(path), 'row': index, 'original_id': row.get('case_id', row.get('sample_id'))}
            result.append(semantic_record(dataset, row, locator))
    return dataset, total, result


def stage_text(output, workers):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, p) for ds, root in DATASETS.items() if ds != 'derived'
            for p in sorted(root.rglob('*.parquet'))]
    jobs.append(('derived', DATASETS['derived'] / 'manifest.jsonl'))
    target = output / 'semantic_sources.jsonl'
    accepted_counts, published = Counter(), Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, target.with_suffix('.tmp').open('w') as stream:
        for ds, total, rows in pool.map(text_shard, jobs):
            published[ds] += total
            accepted_counts[ds] += len(rows)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    target.with_suffix('.tmp').replace(target)
    write_json(output / 'semantic_inventory.json', {'accepted': dict(accepted_counts),
               'published': dict(published), 'sha256': file_hash(target), 'training_ready': False})


def materialize(dataset, row, locator, output):
    uid = source_id(dataset, locator)
    folder = output / 'assets' / dataset / uid[-2:] / uid
    if dataset == 'derived':
        source = str(DATASETS[dataset] / row['source_image'])
        target = str(DATASETS[dataset] / row['edited_image'])
        instruction, native = row['editing_instruction'], row['task_type']
        instances = [{'instance_id': 'selected_region', 'ref': row.get('region_contract', {}).get('segmentation_target'),
                      'rle_size': row['mask_rle']['size'], 'rle_counts': row['mask_rle']['counts']}]
        ground = {'region_contract': row.get('region_contract'), 'planning_details': row.get('planning_details')}
        group = [dataset, row['source_subset'], row['parquet_row_index']]
        mask_path = None
    else:
        sk, tk = {'refedit': ('source_img', 'target_img'), 'crispedit': ('input_img', 'output_img'),
                  'scaleedit': ('source_image', 'edited_image')}[dataset]
        source, target = row[sk], row[tk]
        instruction = row.get('final_instruction') or row['instruction']
        native = row.get('final_task', row.get('type'))
        prefix = '' if dataset == 'refedit' else 'mask__'
        instances = [{k: item.get(k) for k in ('instance_id', 'ref', 'grounding_image', 'mapped_from_target',
                                               'rle_size', 'rle_counts', 'edit_id', 'change_id')}
                     for item in row[prefix + 'instance_masks']]
        ground = parse_json(row.get(prefix + 'ground_json'))
        mask_path, _, _, _ = image_asset(row[prefix + 'mask_png'], folder / 'mask.png')
        group = [dataset, row.get('img_id', row.get('sample_id', locator))]
    src, source_size, source_mode, source_hash = image_asset(source, folder / 'source.img')
    dst, target_size, target_mode, target_hash = image_asset(target, folder / 'target.img')
    # Only file/coordinate readability, not mask area, union, or semantic QC.
    if not instances:
        raise ValueError(f'{uid}: missing dataset instance masks')
    for instance in instances:
        if not instance.get('rle_counts') or instance.get('rle_size') != source_size[::-1]:
            raise ValueError(f'{uid}: unreadable or wrong-coordinate instance RLE')
    return {'id': uid, 'dataset': dataset, 'locator': locator, 'native_type': native,
            'provisional_type': native if dataset == 'derived' else native_edit_type(row),
            'instruction': instruction, 'edit_image': src, 'image': dst,
            'source_size': source_size, 'target_size': target_size,
            'source_mode': source_mode, 'target_mode': target_mode,
            'image_sha256': [source_hash, target_hash], 'split_group': group,
            'dataset_mask': mask_path, 'instances': instances, 'grounding': ground,
            'quality_accepted': True, 'training_ready': False}


def stage_shard(job):
    dataset, path, output = job
    shard_key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
    dest = output / 'source_shards' / f'{dataset}-{shard_key}.jsonl'
    receipt = dest.with_suffix('.receipt.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['manifest_sha256'] != file_hash(dest):
            raise ValueError(f'Corrupt staged shard: {dest}')
        return saved
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    total, kept = 0, 0
    if dataset == 'derived':
        # This dataset is one large manifest, unlike the 3,022 parquet shards.
        # Parallelize its image decoding too; otherwise one thread reads all
        # 55,914 image paths while the other shard workers are idle.
        with path.open() as source:
            records = list(enumerate(json.loads(line) for line in source))
        total = len(records)
        def build(item):
            index, row = item
            locator = {'file': str(path), 'row': index, 'original_id': row.get('case_id', row.get('sample_id'))}
            return materialize(dataset, row, locator, output)
        with ThreadPoolExecutor(max_workers=8) as pool, tmp.open('w') as stream:
            for staged in pool.map(build, (item for item in records if accepted(dataset, item[1]))):
                stream.write(json.dumps(staged, ensure_ascii=False) + '\n')
                kept += 1
    else:
        records = (r for batch in pq.ParquetFile(path).iter_batches(batch_size=8) for r in batch.to_pylist())
        with tmp.open('w') as stream:
            for index, row in enumerate(records):
                total += 1
                if not accepted(dataset, row):
                    continue
                locator = {'file': str(path), 'row': index, 'original_id': row.get('case_id', row.get('sample_id'))}
                staged = materialize(dataset, row, locator, output)
                stream.write(json.dumps(staged, ensure_ascii=False) + '\n')
                kept += 1
    tmp.replace(dest)
    saved = {'dataset': dataset, 'file': str(path), 'total': total, 'accepted': kept,
             'manifest': str(dest), 'manifest_sha256': file_hash(dest)}
    write_json(receipt, saved)
    print(json.dumps({'dataset': dataset, 'accepted': kept, 'shard': str(path)}), flush=True)
    return saved


def stage(output, workers):
    output = Path(output).resolve()
    jobs = [(ds, p, output) for ds, root in DATASETS.items() if ds != 'derived'
            for p in sorted(root.rglob('*.parquet'))]
    jobs.append(('derived', DATASETS['derived'] / 'manifest.jsonl', output))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(stage_shard, jobs))
    counts, published = Counter(), Counter()
    manifest = output / 'sources.jsonl'
    with manifest.with_suffix('.tmp').open('w') as stream:
        for item in receipts:
            counts[item['dataset']] += item['accepted']
            published[item['dataset']] += item['total']
            with open(item['manifest']) as shard:
                for line in shard:
                    stream.write(line)
    manifest.with_suffix('.tmp').replace(manifest)
    write_json(output / 'source_inventory.json', {'accepted': dict(counts), 'published': dict(published),
               'total_accepted': sum(counts.values()), 'source_manifest_sha256': file_hash(manifest),
               'training_ready': False, 'status': 'images_decoded_and_source_masks_preserved'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--text-only', action='store_true')
    args = p.parse_args()
    (stage_text if args.text_only else stage)(args.output, args.workers)


if __name__ == '__main__':
    main()
