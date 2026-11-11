"""v6 second-stage training rows (2026-11-08), built from the v4 / v5 second-stage rows so the structure that produced
v4's execution is kept: 10% grounding rows, 80% token edits, 10% plain edits, the same update slices and order, the
same multi-target (twin) pairs.

Four changes, each from a measured result of the 2026-11-07 arms:
  1. **Plain-edit rows that pointed at NHR-Edit are replaced.** NHR-Edit was deleted on 2026-11-07 at the user's
     request, and 25,490 of the 51,240 plain rows referenced it (199 of a 400-row sample had no source file), so the
     old rows cannot be trained as they are. Replacements come from the self-distilled plain pool (audited pass only).
  2. **Composite : paste-back = 70 : 30 among the non-twin single-target edits** (was 27.1%). mix70 was the best of
     100:0 / 70:30 / 50:50 / 0:100 on MIRA Overall with the highest PQ, and 30% paste-back costs no quality.
  3. **Every twin row keeps its paste-back target** (8,712 of 49,672 carried a composite). The original S1 kept twin
     rows on paste-back deliberately; S1-A overrode that and its twin binding collapsed.
  4. Grounding rows stay the ORIGINAL overlay highlight (no style field), as in S1-B: colour splash cost PF 0.84 and
     IF-Bench S4-noref 0.09 at the same stage, with no quality gain.
The composite pool is v2_final + v3_final, audited-pass only (26,643 + 27,045 = 53,688 pairs).

usage: build_rows_v6s2.py [--out rows_v6_s2.jsonl] [--updates 16000] [--seed 20261108]
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

Q = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/quality_anchor_20261104")
P = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SITEdit_SelfDistill_20261104")
HL = ("hl", "hl_ref", "hlm")
HQ_SHARE = 0.70


def pasteback(r):
    return {k: v for k, v in r.items() if k not in ("target_override", "change_mask")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rows_v6_s2.jsonl")
    ap.add_argument("--updates", type=int, default=16000)
    ap.add_argument("--rows-per-update", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    N = a.updates * a.rows_per_update
    rng = random.Random(a.seed)

    base = []
    with open(Q / "data_round2_v5/rows_v5_s2.jsonl") as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            base.append(json.loads(line))
    assert len(base) == N, f"rows_v5_s2.jsonl holds {len(base)} rows, need {N}"

    # pools: audited-pass composites and audited-pass plain rows, from both batches
    comp, plain = [], []
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
        for l in open(d / "pool_plain.jsonl"):
            r = json.loads(l)
            if r["id"].removeprefix("sd:") in ok:
                plain.append({"kind": "edit_plain", "id": r["id"], "source": r["source"], "target": r["target"],
                              "mask": r["mask"], "prompt": r["prompt"]})
    rng.shuffle(comp); rng.shuffle(plain)
    print(f"pools: composites {len(comp)}, plain {len(plain)}")

    # 1. plain rows that reference the deleted NHR-Edit
    nhr = [i for i, r in enumerate(base) if r["kind"] == "edit_plain" and any("NHR-Edit" in str(v) for v in r.values())]
    for k, i in enumerate(nhr):
        base[i] = plain[k % len(plain)]
    print(f"replaced {len(nhr)} NHR plain rows with self-distilled plain rows")

    # 2/3. twin rows -> paste-back; non-twin single-target edits -> 70% composite
    twins = [i for i, r in enumerate(base) if r.get("pair_kind") == "twin"]
    stripped = 0
    for i in twins:
        if base[i].get("target_override"):
            base[i] = pasteback(base[i]); stripped += 1
    single = [i for i, r in enumerate(base)
              if r["kind"] == "edit_merged" and r.get("pair_kind") != "twin" and not r.get("partners")]
    have = [i for i in single if base[i].get("target_override")]
    want = round(HQ_SHARE * len(single))
    need = want - len(have)
    pick = [i for i in single if not base[i].get("target_override")]
    rng.shuffle(pick)
    for k, i in enumerate(pick[:max(0, need)]):
        base[i] = comp[k % len(comp)]
    print(f"twin rows {len(twins)}, composites stripped {stripped}")
    print(f"single-target edits {len(single)}: composite {len(have)} -> {len(have) + max(0, need)} "
          f"({(len(have) + max(0, need)) / len(single):.1%})")

    out = Q / "data_round2_v6p" / a.out
    with open(out, "w") as f:
        for r in base:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(r["kind"] for r in base)
    g = sum(v for k, v in c.items() if k in HL)
    print(f"{out.name}: {len(base)} rows | {dict(c)}")
    print(f"  grounding {g} ({g / len(base):.1%}) | edit_merged {c['edit_merged']} ({c['edit_merged'] / len(base):.1%})"
          f" | edit_plain {c['edit_plain']} ({c['edit_plain'] / len(base):.1%})")
    print(f"  twin rows {sum(1 for r in base if r.get('pair_kind') == 'twin')}, of which composite "
          f"{sum(1 for r in base if r.get('pair_kind') == 'twin' and r.get('target_override'))}")
    print(f"  rows with a style field (must be 0): {sum(1 for r in base if r.get('style'))}")
    print(f"  NHR references left (must be 0): {sum(1 for r in base if any('NHR-Edit' in str(v) for v in r.values()))}")


if __name__ == "__main__":
    main()
