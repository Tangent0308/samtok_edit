"""ARNOLD text annotation: 4 nodes x 8 independent vLLM replicas, no DDP."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .annotate_full import compact_source, validate_annotation
from .annotate_full import verify_semantic_review
from .cluster import Pipeline, atomic_json, record_failure
from .data import file_hash, row_hash, write_json


def verify_prepared_data(data_root, semantic_sources, semantic_sha256):
    """Bind text conversion to the already prepared image/mask source manifest."""
    data_root = Path(data_root).resolve()
    semantic_sources = Path(semantic_sources).resolve()
    image_sources = data_root / 'sources.jsonl'
    semantic_inventory = json.loads((data_root / 'semantic_inventory.json').read_text())
    source_inventory = json.loads((data_root / 'source_inventory.json').read_text())
    if semantic_sources != data_root / 'semantic_sources.jsonl':
        raise ValueError('Semantic input must be the manifest in the prepared data directory')
    if semantic_sha256 != semantic_inventory['sha256']:
        raise ValueError('Prepared semantic manifest differs from its inventory')
    if file_hash(image_sources) != source_inventory['source_manifest_sha256']:
        raise ValueError('Prepared image/mask manifest differs from its inventory')
    if semantic_inventory['accepted'] != source_inventory['accepted']:
        raise ValueError('Text and image/mask inventories have different dataset counts')
    count = 0
    with semantic_sources.open() as text_stream, image_sources.open() as image_stream:
        for count, (text_line, image_line) in enumerate(zip_longest(text_stream, image_stream), 1):
            if text_line is None or image_line is None:
                raise ValueError(f'Text and image/mask manifest lengths differ at row {count}')
            text_row, image_row = json.loads(text_line), json.loads(image_line)
            for key in ('id', 'dataset', 'instruction', 'provisional_type'):
                if text_row[key] != image_row[key]:
                    raise ValueError(f'Text/image manifest {key} mismatch at row {count}')
    if count != source_inventory['total_accepted']:
        raise ValueError(f'Prepared row count {count} differs from inventory')
    return {'data_root': str(data_root), 'semantic_sources': str(semantic_sources),
            'semantic_sha256': semantic_sha256, 'image_sources': str(image_sources),
            'image_sources_sha256': source_inventory['source_manifest_sha256'],
            'accepted_by_dataset': source_inventory['accepted'], 'input_count': count,
            'image_and_mask_assets': 'referenced by sources.jsonl', 'training_ready': False}


def merge(sources, shards, output, world_size):
    """Join by immutable IDs; incomplete/duplicate shard ownership is an error."""
    output, shards = Path(output), Path(shards)
    with Path(sources).open() as stream:
        inputs = [json.loads(line) for line in stream]
    if not inputs:
        raise ValueError('Empty semantic source manifest')
    if len({r['id'] for r in inputs}) != len(inputs):
        raise ValueError('Duplicate source IDs')
    results, identities = {}, []
    for rank in range(world_size):
        path = shards / f'annotations-{rank:02d}.jsonl'
        complete = json.loads((shards / f'complete-{rank:02d}.json').read_text())
        if complete['sha256'] != file_hash(path):
            raise ValueError(f'Shard {rank} changed after completion')
        identity = json.loads((shards / f'identity-{rank:02d}.json').read_text())
        if identity['shard'] != rank or identity['shards'] != world_size:
            raise ValueError('Wrong shard identity')
        identities.append({k: v for k, v in identity.items() if k != 'shard'})
        owned = {r['id'] for r in inputs[rank::world_size]}
        latest = {}
        with path.open() as stream:
            lines = stream.readlines()
        for line in lines:
            value = json.loads(line)
            if value['id'] not in owned:
                raise ValueError(f'Shard {rank} emitted an unowned ID')
            if value['id'] in latest and latest[value['id']]['status'] == 'accepted':
                raise ValueError('Duplicate accepted record during resume')
            latest[value['id']] = value
        if set(latest) != owned:
            raise ValueError(f'Shard {rank} did not process every assigned input')
        results.update(latest)
    if any(x != identities[0] for x in identities):
        raise ValueError('Annotation worker identities disagree')
    if identities[0]['sources_sha256'] != file_hash(sources):
        raise ValueError('Merge sources differ from worker input')
    accepted_counts, failures, kinds = Counter(), [], Counter()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'annotations.jsonl'
    with destination.with_suffix('.tmp').open('w') as stream:
        for source in inputs:
            value = results[source['id']]
            if value['status'] != 'accepted':
                failures.append({'source': source, 'result': value})
                continue
            if value['source_sha256'] != row_hash(compact_source(source)):
                raise ValueError('Annotation source content changed')
            validate_annotation(source, value['annotation'])
            if value['review'].get('valid') is not True:
                raise ValueError('Accepted annotation has no positive protocol checks')
            verify_semantic_review(source, value['annotation'], value['review'])
            accepted_counts[source['dataset']] += 1
            kinds['composite' if len(value['annotation']['units']) > 1 else
                  value['annotation']['units'][0]['edit_type']] += 1
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    destination.with_suffix('.tmp').replace(destination)
    with (output / 'failed.jsonl').open('w') as stream:
        for value in failures:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    report = {'input_count': len(inputs), 'accepted_count': sum(accepted_counts.values()),
              'accepted_by_dataset': dict(accepted_counts), 'failed_count': len(failures),
              'edit_types': dict(kinds), 'identity': identities[0],
              'annotations_sha256': file_hash(destination), 'candidates_complete': not failures,
              'semantic_ready': False,
              'training_ready': False, 'review_kind': 'two-field-generation-plus-deterministic-protocol-checks'}
    write_json(output / 'conversion_report.json', report)
    if not failures:
        write_json(output / 'CANDIDATES_COMPLETE.json', report)
    return report


class AnnotationPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.env['OMP_NUM_THREADS'] = '4'
        self.children = []

    def run(self):
        args = self.args
        sources = Path(args.sources).resolve()
        source_hash = file_hash(sources)
        local = Path(args.local_root) / self.root.name / f'node{self.rank}'
        local.mkdir(parents=True, exist_ok=False)
        local_sources = local / 'sources.jsonl'
        shutil.copyfile(sources, local_sources)
        if file_hash(local_sources) != source_hash:
            raise ValueError('Local input copy corrupted')
        if args.prepared_data_root and self.rank == 0:
            atomic_json(self.root / 'input_linkage.json',
                        verify_prepared_data(args.prepared_data_root, sources, source_hash))
        model = local / 'model'
        shutil.copytree(args.model, model)
        model_identity = {str(p.relative_to(model)): file_hash(p) for p in sorted(model.iterdir())
                          if p.suffix in {'.json', '.safetensors', '.txt', '.jinja'}}
        write_json(local / 'model-identity.json', model_identity)
        common = {'nodes': self.topo['nodes'], 'gpus_per_node': self.topo['gpus'],
                  'shards': self.topo['world_size'], 'sources_sha256': source_hash,
                  'prepared_data_root': str(Path(args.prepared_data_root).resolve())
                  if args.prepared_data_root else None,
                  'model': model_identity, 'git_commit': self.git_commit,
                  'code': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                  'batch_size': args.batch_size, 'attempts': args.attempts,
                  'max_model_len': args.max_model_len, 'gpu_memory_utilization': args.gpu_memory_utilization,
                  'resume_from': str(Path(args.resume_from).resolve()) if args.resume_from else None,
                  'packages': {k: importlib.metadata.version(k) for k in
                               ('vllm', 'torch', 'transformers')}}
        atomic_json(self.node / 'topology.json', common)
        self.barrier('topology')
        for rank in range(self.topo['nodes']):
            if json.loads((self.root / 'nodes' / str(rank) / 'topology.json').read_text()) != common:
                raise ValueError(f'Annotation input/code/model/environment mismatch on node {rank}')
        if self.rank == 0:
            atomic_json(self.root / 'manifest.json', common)
        shards = self.root / 'shards'
        shards.mkdir(exist_ok=True)
        if args.resume_from:
            previous = Path(args.resume_from) / 'shards'
            for gpu in range(self.topo['gpus']):
                rank = self.rank * self.topo['gpus'] + gpu
                for name in (f'annotations-{rank:02d}.jsonl', f'identity-{rank:02d}.json'):
                    old = previous / name
                    if old.exists():
                        temporary = shards / (name + '.tmp')
                        shutil.copyfile(old, temporary)
                        temporary.replace(shards / name)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        visible = [x.strip() for x in visible if x.strip()] or [str(i) for i in range(8)]
        if len(visible) != self.topo['gpus']:
            raise ValueError('Expected eight visible GPUs')
        handles = []
        try:
            for gpu in range(self.topo['gpus']):
                rank = self.rank * self.topo['gpus'] + gpu
                command = [sys.executable, '-m', 'samtok_edit21.annotate_full', '--sources', str(local_sources),
                           '--output', str(shards), '--model', str(model), '--model-identity', str(local / 'model-identity.json'),
                           '--shard', str(rank), '--shards', str(self.topo['world_size']),
                           '--batch-size', str(args.batch_size), '--attempts', str(args.attempts),
                           '--max-model-len', str(args.max_model_len),
                           '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
                stream = (self.logdir / f'annotation-{rank:02d}.log').open('w')
                handles.append(stream)
                env = dict(self.env, CUDA_VISIBLE_DEVICES=visible[gpu],
                           VLLM_CACHE_ROOT=str(local / f'vllm-cache-{gpu}'))
                process = subprocess.Popen(command, env=env, cwd=self.repo, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(process)
                atomic_json(self.node / f'gpu{gpu}.command.json', command)
            deadline, next_log = time.monotonic() + args.timeout, 0
            while any(p.poll() is None for p in self.children):
                self.check_failures()
                if any(p.poll() not in (None, 0) for p in self.children):
                    raise RuntimeError(f'Annotation worker failed; see {self.logdir}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Annotation phase timed out')
                if self.rank == 0 and time.monotonic() >= next_log:
                    progress = [json.loads(p.read_text()) for p in shards.glob('progress-*.json')]
                    status = {'accepted': sum(p['accepted'] for p in progress),
                              'failed': sum(p['failed_this_attempt'] for p in progress),
                              'reporting_shards': len(progress)}
                    print(json.dumps(status), flush=True)
                    next_log = time.monotonic() + 30
                time.sleep(2)
            if any(p.returncode != 0 for p in self.children):
                raise RuntimeError('Annotation child failed')
            self.barrier('annotation')
            if self.rank == 0:
                report = merge(sources, shards, self.root, self.topo['world_size'])
                print(json.dumps({k: report[k] for k in
                      ('input_count', 'accepted_count', 'failed_count', 'candidates_complete')}), flush=True)
            self.barrier('merge')
            if self.rank == 0:
                atomic_json(self.root / 'SUCCESS.json', {'time': time.time(),
                                                        'processed_complete': True,
                                                        'input_count': report['input_count'],
                                                        'accepted_count': report['accepted_count'],
                                                        'failed_count': report['failed_count'],
                                                        'candidates_complete': report['candidates_complete'],
                                                        'semantic_ready': False,
                                                        'training_ready': False})
        finally:
            for process in self.children:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in self.children:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            for stream in handles:
                stream.close()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', required=True)
    p.add_argument('--prepared-data-root',
                   help='Prepared data directory with aligned semantic_sources.jsonl and sources.jsonl')
    p.add_argument('--run-root', required=True)
    p.add_argument('--model', default='/mnt/bn/strategy-mllm-train/common/models/Qwen3-4B-Instruct-2507')
    p.add_argument('--local-root', default='/tmp/samtok-annotation')
    p.add_argument('--resume-from', help='Previous run directory; must have identical code/model/input/sharding')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.75)
    p.add_argument('--timeout', type=int, default=86400)
    p.add_argument('--local', action='store_true')
    args = p.parse_args()
    pipeline = AnnotationPipeline(args)
    try:
        pipeline.run()
    except BaseException as exc:
        record_failure(pipeline.node / 'failure.json', {'error': str(exc), 'type': type(exc).__name__, 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
