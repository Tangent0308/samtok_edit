"""Fail closed on missing ranks, wrong ratios, gradients, A/C, cache, W&B or inference."""
import argparse,json,math
from collections import Counter
from pathlib import Path
import torch
from safetensors.torch import load_file
from samtok_edit21.data import file_hash,write_json
from samtok_edit21.provenance import verify_cache
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();root=Path(a.run_root)
manifest=json.loads((root/'manifest.json').read_text());world=manifest['world_size'];config=manifest['args'];report={}
def lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
for rank in range(world):
    probe=json.loads((root/'collectives'/f'rank{rank}.json').read_text());assert probe['rank']==rank and probe['world_size']==world and probe['all_reduce']==world*(world+1)/2
for stage,accum,ratio in [('stage1',8,{'edit_ntp':3,'edit_umt:ref':2,'edit_umt:noref':2,'edit':1}),('stage2',4,{'edit_umt:ref':1,'edit_umt:noref':2,'edit':1})]:
    directory=root/stage;steps=config[f'{stage}_steps'];updates=lines(directory/'optimizer_steps.jsonl');metrics=lines(directory/'training_metrics.jsonl');assert len(updates)==len(metrics)==steps
    for i,entry in enumerate(metrics,1):
        assert entry['optimizer_step']==i and not entry['skipped'] and entry['samples']==world*accum
        assert entry['rank_samples']=={str(r):accum for r in range(world)}
        assert entry['branches']=={k:v*world for k,v in ratio.items()}
        assert all(math.isfinite(v) for v in entry['metrics'].values())
        assert abs(entry['metrics']['region_weight_sum']-1)<1e-5
        assert entry['counts']['region_inside_mse']==world*(4 if stage=='stage1' else 3)
        if stage=='stage2':
            assert abs(entry['metrics']['attention_weight']-(0 if i==1 else config['attention_weight']))<1e-7
            assert entry['counts']['attn_main']==world*3
            assert entry['metrics']['attn_main']>=0 and entry['metrics']['attn_read']>=0
    for rank in range(world):
        grads=lines(directory/f'gradients-rank{rank}.jsonl');assert len(grads)==steps*accum,(stage,rank,len(grads))
        assert Counter(g['branch'] for g in grads)=={k:v*steps for k,v in ratio.items()}
        assert all(math.isfinite(g['grad_norm_before_clip']) and g['grad_norm_before_clip']>0 and g['current_backward_grad_peak']>0 and g['frozen_grad_tensors']==0 for g in grads)
    parameters=json.loads((directory/'rank_parameters.json').read_text());assert len(parameters)==world and {r['rank'] for r in parameters}==set(range(world))
    assert len({r['sha256'] for r in parameters})==1 and len({r['trainable_parameters'] for r in parameters})==1
    adapter=load_file(str(directory/'adapter/adapter.safetensors'));assert all('lora_' in k and torch.isfinite(v).all() for k,v in adapter.items())
    assert any(v.count_nonzero().item()>0 for k,v in adapter.items() if 'lora_B' in k)
    wb=json.loads((directory/'wandb.json').read_text());assert wb['status']=='finished' and wb['mode']==config['wandb_mode']
    report[stage]={'rank_parameters':parameters,'updates':steps,'microsteps_per_rank':steps*accum,'all_rank_gradients':'finite/nonzero; frozen grads absent','ratios':'exact on every rank','adapter_sha256':file_hash(directory/'adapter/adapter.safetensors'),'wandb':wb,'final_metrics':metrics[-1]}
cache=json.loads((root/'cache/manifest.json').read_text());verify_cache(root/'cache',cache);assert cache['row_count']==54
shards=Counter(int(r['_cache_file'].split('/')[0]) for r in cache['rows']);assert set(shards)==set(range(world))
assert cache['identity']['te_adapter']['sha256']==report['stage1']['adapter_sha256']
report['cache']={'rows':54,'per_rank_shards':dict(shards),'verified':True,'supervision_identity':cache['supervision_identity']}
images=[]
for rank in range(8):
    image=json.loads((root/'inference'/f'rank{rank}.json').read_text());assert image['finite'] and image['size']==[256,256] and image['image_mode']=='RGBA';images.append(image)
report.update(world_size=world,inference=images,passed=True)
write_json(root/'audit.json',report);print(json.dumps({'passed':True,'world_size':world,'stage1_updates':config['stage1_steps'],'stage2_updates':config['stage2_steps'],'cache_rows':54,'inference_gpus':8}))
