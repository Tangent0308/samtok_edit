"""Specs for self-distillation v3 (2026-11-07, PLAN_V5.md section 13): the second batch. Single-unit records that S1/S2
train on and that no earlier batch (pilot, ext, v2) generated, ScaleEdit included (licence confirmed by the user on
2026-11-07), all colour/material attribute and add edits first, then replace/remove. Same generator, gates and audit as v2."""
import collections, json, random, sys
from pathlib import Path
import pandas as pd
ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")
S = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/round2_scaleup_20261030")
P = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SITEdit_SelfDistill_20261104")
out, total, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed)
def ids(path):
    s = []
    for l in open(path):
        r = json.loads(l)
        if r["kind"] == "edit_merged" and "order" not in r and "forms" not in r:
            s.append(r["id"])
    return s
s1 = list(dict.fromkeys(ids(S / "rows_s1.jsonl"))); s2 = list(dict.fromkeys(ids(S / "rows_s2.jsonl")))
done = set()
for d in ("pilot", "ext", "v2"):
    for p in (P / d / "shards").glob("shard_*.jsonl"):
        for l in open(p):
            done.add(json.loads(l)["id"])           # every attempted id (completed or not) of the earlier batches
done |= {json.loads(l)["id"] for l in open(P / "v2/specs.jsonl")}
u = pd.read_parquet(ROOT / "units.parquet")
u = u[~u["multi"]].drop_duplicates("record")
info = {r: (o, op, tr, a) for r, o, op, tr, a in zip(u["record"], u["origin"], u["op"], u["text_role"], u["area"])}
def ok(rid):
    if rid in done or rid not in info: return False
    o, op, tr, _ = info[rid]
    return op in ("attribute", "replace", "add", "remove") and not (isinstance(tr, str) and tr)
first = [r for r in s1 if ok(r)]
first_set = set(first)
rest = [r for r in s2 if r not in first_set and ok(r)]
# v3: every colour/material attribute record first, then every add, then replace/remove in random order (v2 used a
# random weighting that left a quarter of the attribute records out)
priority = {"attribute": 0, "add": 1, "replace": 2, "remove": 2}
rng.shuffle(rest)
rest.sort(key=lambda r: priority[info[r][1]])
pick = (first + rest)[:total]
want = set(pick); recs = {}
with open(ROOT / "records.jsonl") as f:
    for line in f:
        rid = line[8:line.index('"', 8)]
        if rid in want:
            recs[rid] = json.loads(line)
            if len(recs) == len(want): break
rows = []
for rid in pick:
    r = recs[rid]; x = r["units"][0]; spec = x["instruction_spec"]
    if not x.get("instruction_ref"): continue
    rows.append({"id": r["id"], "family": r["family"], "origin": r["origin"], "op": x["op"], "area": float(info[rid][3]), "group": r["group"],
                 "prompt": x["instruction_ref"], "property": [a.get("property") for a in spec.get("attributes", [])],
                 "source_image": r["source_image"], "size": r["size"], "in_s1": rid in first_set})
rng.shuffle(rows)          # shards get a balanced mix
out.mkdir(parents=True, exist_ok=True)
(out / "specs.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
print("S1 ids", len(s1), "eligible", len(first), "| S2 singles", len(s2), "eligible rest", len(rest), "| written", len(rows), "| already generated (reused)", len(done))
print("by op", collections.Counter(x["op"] for x in rows)); print("by family", collections.Counter(x["family"] for x in rows)); print("by origin", collections.Counter(x["origin"] for x in rows))
