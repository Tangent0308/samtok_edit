"""v5 training rows, built from the v4 S1/S2 rows (same update slices, same order, same twins/multi-target structure).

S1' (per update of 32 rows): ~60% highlight (rendered as tint at training time), ~30% token edits with high-quality
     targets, ~10% plain edits (no token).  The original single-unit edit rows get their self-distilled composite when
     one exists; flat ones without it are replaced by a high-quality pool row; highlight rows above the 60% share become
     high-quality token rows or plain rows.
S2' (per update): highlight rows kept (~10%); multi-target rows unchanged; single-unit edit rows get their composite
     when one exists, flat ones without it are replaced; ~10% become plain rows; further non-twin rows are replaced by
     pool rows until high-quality targets are at least --hq-share of the single-unit edit rows.
High-quality token rows: {"kind": "edit_merged", "id", "form", "target_override", "change_mask"} (pool of finalize_v2.py,
only items whose semantic audit passed when --audit is given).  Plain rows: {"kind": "edit_plain", ...} from NHR-Edit
(category-capped, at most 2 per source image) and the self-distilled plain pool, half each.

usage: build_rows_v5.py --stage s1|s2 --rows <v4 rows> --out <rows file> --token-pool P [--audit A] --nhr <pool parts dir>
       --sd-plain <pool_plain.jsonl> --flat <flat_ids.txt> [--seed 20261105]
"""
import argparse, collections, json, random
from pathlib import Path

HL = ("hl", "hl_ref", "hlm")
ADD_REMOVE = {"Add object", "Remove object", "Add object and Remove object"}


def stochastic(x, rng):
    base = int(x)
    return base + (rng.random() < x - base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("s1", "s2"), required=True)
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--token-pool", required=True); ap.add_argument("--audit")
    ap.add_argument("--nhr", required=True); ap.add_argument("--sd-plain", required=True)
    ap.add_argument("--flat", required=True)
    ap.add_argument("--hl-share", type=float, default=0.60); ap.add_argument("--plain-share", type=float, default=0.10)
    ap.add_argument("--hq-share", type=float, default=0.35); ap.add_argument("--nhr-add-remove-cap", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    rng = random.Random(a.seed + (1 if a.stage == "s2" else 0))

    pool = {json.loads(l)["id"]: json.loads(l) for l in open(a.token_pool)}
    if a.audit:
        passed = {json.loads(l)["id"] for l in open(a.audit) if json.loads(l).get("pass")}
        pool = {k: v for k, v in pool.items() if k in passed}
    pool_ids = sorted(pool); rng.shuffle(pool_ids)
    flat = set(Path(a.flat).read_text().split())

    nhr = [json.loads(l) for p in sorted(Path(a.nhr).glob("*.jsonl")) for l in open(p)]
    rng.shuffle(nhr)
    per_seed = collections.Counter(); change, addrem = [], []
    for r in nhr:
        if per_seed[r["seed"]] >= 2:
            continue
        per_seed[r["seed"]] += 1
        (addrem if r["category"] in ADD_REMOVE else change).append(r)
    cap = int(len(change) * a.nhr_add_remove_cap / (1 - a.nhr_add_remove_cap))
    nhr = change + addrem[:cap]; rng.shuffle(nhr)
    sd_plain = [json.loads(l) for l in open(a.sd_plain)]
    if a.audit:
        sd_plain = [r for r in sd_plain if r["id"][3:] in pool]
    rng.shuffle(sd_plain)
    counters = collections.Counter(); used_hq = collections.Counter(); ptr = {"hq": 0, "nhr": 0, "sd": 0}

    def hq_row(entry_id=None):
        if entry_id is None:
            entry_id = pool_ids[ptr["hq"] % len(pool_ids)]; ptr["hq"] += 1
        e = pool[entry_id]; used_hq[entry_id] += 1
        form = rng.choices(["noref_token", "ref_token"], [0.7, 0.3])[0] if e.get("noref", True) else "ref_token"
        return {"kind": "edit_merged", "id": entry_id, "form": form, "target_override": e["target"], "change_mask": e["mask"]}

    def plain_row():
        use_sd = sd_plain and (not nhr or rng.random() < 0.5)
        if use_sd:
            r = sd_plain[ptr["sd"] % len(sd_plain)]; ptr["sd"] += 1; counters["plain_sd"] += 1
            prompt = r["prompt"]
        else:
            r = nhr[ptr["nhr"] % len(nhr)]; ptr["nhr"] += 1; counters["plain_nhr"] += 1
            prompt = rng.choice(r["augmented"]) if r.get("augmented") and rng.random() < 0.5 else r["prompt"]
        return {"kind": "edit_plain", "id": str(r["id"]), "source": r["source"], "target": r["target"], "mask": r.get("mask"), "prompt": prompt}

    def upgrade(row):
        """Composite for a single-unit edit row when one exists; flat rows without one are replaced."""
        if row["id"] in pool:
            used_hq[row["id"]] += 1; counters["substituted"] += 1
            return dict(row, target_override=pool[row["id"]]["target"], change_mask=pool[row["id"]]["mask"])
        if row["id"] in flat:
            counters["flat_replaced"] += 1
            return hq_row()
        return row

    rows = [json.loads(l) for l in open(a.rows)]
    assert len(rows) % 32 == 0
    out = []
    for u in range(0, len(rows), 32):
        s = [dict(x) for x in rows[u:u + 32]]
        single = lambda x: x["kind"] == "edit_merged" and "order" not in x and "forms" not in x  # noqa: E731
        for i, x in enumerate(s):
            if single(x):
                s[i] = upgrade(x)
        n_plain = stochastic(32 * a.plain_share, rng)
        if a.stage == "s1":
            hl_idx = [i for i, x in enumerate(s) if x["kind"] in HL]
            rng.shuffle(hl_idx)
            extra = max(0, len(hl_idx) - stochastic(32 * a.hl_share, rng))
            for j, i in enumerate(hl_idx[:extra]):
                if j < n_plain:
                    s[i] = plain_row()
                else:
                    s[i] = hq_row(); counters["hq_from_highlight"] += 1
            # S1' edit rows should all carry high-quality targets; twin rows keep their (paired) original targets
            for i, x in enumerate(s):
                if single(x) and "partners" not in x and "target_override" not in x:
                    s[i] = hq_row(); counters["hq_topup_s1"] += 1
        else:
            free = [i for i, x in enumerate(s) if single(x) and "partners" not in x and "target_override" not in x]
            rng.shuffle(free)
            for i in free[:n_plain]:
                s[i] = plain_row()
            singles = [i for i, x in enumerate(s) if single(x)]
            hq = sum("target_override" in s[i] for i in singles)
            for i in [i for i in free[n_plain:]]:
                if hq >= a.hq_share * len(singles):
                    break
                s[i] = hq_row(); hq += 1; counters["hq_topup"] += 1
        out += s
    kinds = collections.Counter(x["kind"] for x in out)
    hq_rows = sum("target_override" in x for x in out)
    singles = [x for x in out if x["kind"] == "edit_merged" and "order" not in x and "forms" not in x]
    stats = {"stage": a.stage, "rows": len(out), "updates": len(out) // 32, "kinds": dict(kinds),
             "shares": {k: round(v / len(out), 4) for k, v in kinds.items()},
             "hq_token_rows": hq_rows, "hq_share_of_single_edit_rows": round(sum("target_override" in x for x in singles) / max(len(singles), 1), 4),
             "hq_unique_used": len(used_hq), "hq_pool": len(pool), "hq_max_uses": max(used_hq.values()) if used_hq else 0,
             "plain_pools": {"nhr": len(nhr), "sd": len(sd_plain)}, **counters, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    Path(a.out).with_name(Path(a.out).stem + "_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
