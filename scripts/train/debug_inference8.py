"""Eight independent GPU replicas exercise both trained adapters and inference modes."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from accelerate.utils import set_seed
from samtok_edit21.model import load_pipeline,edit
from samtok_edit21.training import load_adapter
from samtok_edit21.provenance import assert_inference_identity
from samtok_edit21.data import write_json
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--data',required=True);p.add_argument('--qwen',required=True);p.add_argument('--samtok',required=True);a=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);set_seed(100+rank)
root=Path(a.run_root); data=Path(a.data);out=root/'inference';out.mkdir(exist_ok=True)
records=json.loads((data/'provenance.json').read_text());ntp=[json.loads(l) for l in (data/'stage1.jsonl').read_text().splitlines() if json.loads(l)['sample_type']=='edit_ntp']
# Three datasets, all exposed modes, and both oracle variants.
cases=[(0,'direct','ref'),(6,'oracle','ref'),(12,'oracle','noref'),(2,'inline','ref'),(8,'online','ref'),(14,'online','noref'),(9,'oracle','noref'),(16,'direct','ref')]
idx,mode,variant=cases[rank];record=records[idx]
te=root/'stage1/adapter';dit=root/'stage2/adapter'
config=json.loads((dit/'adapter.json').read_text());assert_inference_identity(config['conditioning_identity'],a.qwen,a.samtok,str(te))
pipe=load_pipeline(a.qwen,a.samtok,device=f'cuda:{rank}');load_adapter(pipe.text_encoder,te);load_adapter(pipe.dit,dit);pipe.eval()
prompt=record['instruction']
if mode=='inline':
    rows=[json.loads(l) for l in (data/'stage2.jsonl').read_text().splitlines()]
    prompt=next(r['prompt'] for r in rows if r['edit_image']==record['edit_image'] and r.get('instr_variant')=='ref')
image,report=edit(pipe,prompt,[Image.open(record['edit_image']).convert('RGBA')],mode=mode,variant=variant,
                 cot=ntp[idx]['mt_cot'] if mode=='oracle' else None,reviewed_units=record['units'] if mode=='oracle' or (mode=='online' and variant=='noref') else None,
                 max_new_tokens=192,height=256,width=256,num_inference_steps=4,cfg_scale=1.0,seed=100+rank,use_kv_cache=True)
assert image.size==(256,256) and image.mode=='RGBA'
array=np.asarray(image);assert np.isfinite(array).all() and array[...,:3].std()>0
image.save(out/f'rank{rank}-{mode}.png')
write_json(out/f'rank{rank}.json',{'rank':rank,'dataset':record['source_dataset'],'mode':mode,'variant':variant,'image_mode':image.mode,'size':list(image.size),'report':report,'finite':True,'rgb_std':float(array[...,:3].std()),'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})
print(json.dumps({'rank':rank,'mode':mode,'passed':True,'fallback_reason':report['fallback_reason']}),flush=True)
