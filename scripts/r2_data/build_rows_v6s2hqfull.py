"""Full-size (512,000-row) version of the high-composite second-stage rows (2026-11-08): rows_v6_s2.jsonl with EVERY non-twin
single-target edit carrying its high-quality composite target, so the only paste-back targets left are the twin rows.

Why: the quality a model reaches is bounded by its targets. The first stage trained on composites only (PQ ceiling
7.12 measured by Q2) and reached PQ 7.27, at parity with the base model. The second stage reintroduces paste-back
targets (ceiling 6.49, measured by Q3) at 30.9% of all rows, and the g10 pilot fell to PQ 5.65. This arm keeps the
second stage's 80% edit rows — which is what lifts our own bench (IF-Bench S1 UnitPass 0.667 at the first stage ->
0.780 at g10) — while removing the low-ceiling targets from everything except the twins, which keep paste-back
because that is what preserves twin binding.

Together with g10 (30.9% paste-back) and pb50 (45%), this gives three points of a dose-response curve on both
benches, so the share can be chosen from data instead of guessed.

usage: build_rows_v6s2hq.py      writes data_round2_v6p/rows_v6_s2hqfull.jsonl
"""
import json, random
from collections import Counter
from pathlib import Path

Q = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/quality_anchor_20261104")
P = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SITEdit_SelfDistill_20261104")
D = Q / "data_round2_v6p"
HL = ("hl", "hl_ref", "hlm")

rng = random.Random(20261108)
comp = []
for pool in ("v2_final", "v3_final"):
    d = P / pool
    if not (d / "pool_token.jsonl").is_file():
        continue
    ok = {json.loads(l)["id"] for l in open(d / "audit.jsonl") if json.loads(l).get("pass")}
    for l in open(d / "pool_token.jsonl"):
        t = json.loads(l)
        if t["id"] in ok:
            comp.append({"kind": "edit_merged", "id": t["id"], "form": "noref_token" if t.get("noref") else "ref_token",
                         "target_override": t["target"], "change_mask": t["mask"]})
rng.shuffle(comp)

rows = [json.loads(l) for l in open(D / "rows_v6_s2.jsonl")]
single = [i for i, r in enumerate(rows)
          if r["kind"] == "edit_merged" and r.get("pair_kind") != "twin" and not r.get("partners")]
need = [i for i in single if not rows[i].get("target_override")]
for k, i in enumerate(need):
    rows[i] = comp[k % len(comp)]
out = D / "rows_v6_s2hqfull.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = Counter(r["kind"] for r in rows)
g = sum(v for k, v in c.items() if k in HL)
pb = sum(1 for r in rows if r["kind"] == "edit_merged" and not r.get("target_override"))
tw = sum(1 for r in rows if r.get("pair_kind") == "twin")
print(f"{out.name}: {len(rows)} rows | {dict(c)}")
print(f"  grounding {g} ({g / len(rows):.1%}) | replaced {len(need)} paste-back rows with composites")
print(f"  paste-back rows left {pb} ({pb / len(rows):.1%}), twin rows {tw} (composite "
      f"{sum(1 for r in rows if r.get('pair_kind') == 'twin' and r.get('target_override'))})")
print(f"  style field {sum(1 for r in rows if r.get('style'))}")
base = [json.loads(l) for l in open(D / "rows_v6_s2.jsonl")]
diff = [i for i, (a, b) in enumerate(zip(base, rows)) if a != b]
print(f"  rows differing from g10: {len(diff)} (all of them single-target edit rows: "
      f"{all(base[i]['kind'] == 'edit_merged' and base[i].get('pair_kind') != 'twin' for i in diff)})")
