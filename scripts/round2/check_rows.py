"""Model-free check of a rows file of the two-stage retrain (docs/03): every row kind's regions, negatives, pairs and
prompts as the trainer builds them (training/round2.py), plus a contact sheet of examples (highlights: the target
overlay, negatives in blue; edits: token region green, loss region red, negatives blue, next to the target).
    PYTHONPATH=src:... python scripts/round2/check_rows.py <data dir> <rows file> <sheet.jpg> [max rows]"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training import round2 as r2

TOKEN = re.compile(r"<\|mt_start\|>.*?<\|mt_end\|>")

data, rows_path, sheet = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
rows = [json.loads(l) for l in open(data / rows_path)]
ids = {r["id"] for r in rows if r["kind"] == "edit_merged"} | {p for r in rows for p in r.get("partners", ())}
merged = {}
for line in open(rd.MERGED_ROOT / "records.jsonl"):
    rid = line[8:line.index('"', 8)]
    if rid in ids:
        merged[rid] = json.loads(line)
pool = {}
def entry(sid):
    if sid not in pool:
        pool[sid] = json.load(open(data / "pool" / f"{sid}.json"))
    return pool[sid]
dev = set(json.load(open(data / "dev_split.json"))["sa1b_sources"])
bad, stats, examples = Counter(), Counter(), defaultdict(list)
pairs = defaultdict(list)
for k, row in enumerate(rows[:limit]):
    kind = row["kind"]
    stats[kind] += 1
    if row.get("pair_key"):
        pairs[row["pair_key"]].append((k, row))
    if kind in ("hl", "hl_ref", "hlm"):
        e = entry(row["source_id"])
        if row["source_id"] in dev: bad["hl_on_dev_image"] += 1
        ms = [e["masks"][i] for i in row["masks"]]
        if not all(rd.usable(m) for m in ms): bad["hl_unusable_mask"] += 1
        if len(row["colors"]) != len(ms) or len(row["negatives"]) != len(ms): bad["hl_lengths"] += 1
        for i, negs in zip(row["masks"], row["negatives"]):
            rel = e["masks"][i]["rel"]
            if i in negs or set(negs) & set(rel["duplicate"]) or set(negs) & set(rel["parts"]): bad["hl_negative_is_target_or_part"] += 1
            if set(negs) & set(row["masks"]) and kind == "hlm": bad["hlm_negative_is_other_target"] += 1
        if kind == "hl_ref" and not row.get("phrase"): bad["hl_ref_no_phrase"] += 1
        prompt = r2.highlight_prompt(row, [m["span"] for m in ms], e)
        if prompt.count("<|mt_start|>") != len(ms): bad["hl_prompt_spans"] += 1
        if len(examples[kind]) < 3:
            examples[kind].append((row, e, prompt))
        stats["hl_negatives_nonempty"] += bool(row["negatives"][0])
    elif kind == "edit_merged":
        rec = merged[row["id"]]
        if not rec["default"]: bad["edit_not_default"] += 1
        with Image.open(rd.MERGED_ROOT / rec["source_image"]) as im:
            shape = (im.height, im.width)
        e = entry(rec["sa1b"]["source_id"]) if row.get("negatives") else None
        loss, route, token, sentences, neg, local = r2.merged_regions(
            rec, row, shape, partner_selection=lambda p: r2.rle_region(merged[p]["units"][0]["selection"], shape),
            pool_mask=lambda j: rd.decode_rle(e["masks"][j]["rle"]).astype(bool))
        if any(not m.any() for m in loss): bad["edit_empty_loss"] += 1
        if any(not m.any() for m in token): bad["edit_empty_token"] += 1
        if any(m.shape != shape for m in token + loss + route): bad["edit_shape"] += 1
        if any("<|mt_start|>" not in s for s in sentences): bad["edit_sentence_without_token"] += 1
        if len(sentences) != len(rec["units"]): bad["edit_unit_count"] += 1
        if row.get("form") == "noref_token" and not rec["units"][0].get("instruction_noref"): bad["edit_noref_missing"] += 1
        fam = rec["family"]
        stats[f"edit_{fam}"] += 1
        if row.get("negatives"):
            stats[f"edit_{fam}_with_negatives"] += 1
            if not neg.any(): stats[f"edit_{fam}_negatives_empty_after_mask"] += 1
        for p in row.get("partners", ()):
            pr = merged[p]
            if pr["sa1b"]["source_id"] != rec["sa1b"]["source_id"]: bad["pair_partner_other_image"] += 1
            if rec.get("twin") and rec["twin"]["partner"] != p and fam == "D12": bad["d12_partner_mismatch"] += 1
            sp = r2.rle_region(pr["units"][0]["selection"], shape)
            if not (neg & sp).any() and sp.any(): stats["pair_partner_S_fully_inside_own_region"] += 1
        key = f"{fam}{'_pair' if row.get('partners') else ''}{'_multi' if len(rec['units']) > 1 else ''}"
        if len(examples[key]) < 2:
            examples[key].append((row, rec, sentences, loss, token, neg, shape))
for key, members in pairs.items():
    if len(members) != 2: bad["pair_size"] += 1; continue
    (ka, a), (kb, b) = members
    if a["kind"] in ("hl",):
        if a["template"] != b["template"] or a["colors"] != b["colors"] or a["source_id"] != b["source_id"]: bad["hlpair_not_matched"] += 1
        stats[f"hlpair_{a['pair_kind']}"] += 1
    if ka // 32 != kb // 32: bad["pair_split_over_updates"] += 1
print(json.dumps({"rows": min(len(rows), limit), "kinds": dict(stats), "problems": dict(bad)}, indent=1))

# contact sheet: hl targets (overlay) and edit maps (token green, loss red, negatives blue)
tiles = []
def tile(img, text):
    img = img.convert("RGB"); img.thumbnail((360, 360))
    canvas = Image.new("RGB", (360, 420), "white"); canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for n, line in enumerate([text[i:i + 58] for i in range(0, min(len(text), 290), 58)]):
        d.text((2, 362 + 11 * n), line, fill="black")
    return canvas
for kind in ("hl", "hl_ref", "hlm"):
    for row, e, prompt in examples[kind][:2]:
        with Image.open(e["image"]) as im:
            src = im.convert("RGB")
        regions = [rd.decode_rle(e["masks"][i]["dec"]) for i in row["masks"]]
        tgt = r2.overlay_regions(src, regions, row["colors"])
        a = np.asarray(tgt).copy()
        neg = np.zeros(regions[0].shape, bool)
        for g in row["negatives"]:
            for j in g: neg |= rd.decode_rle(e["masks"][j]["rle"])
        a[neg & ~np.any(regions, 0)] = (0.5 * a[neg & ~np.any(regions, 0)] + [0, 0, 127]).astype(np.uint8)
        tiles.append(tile(Image.fromarray(a), f"{kind} {row.get('pair_kind', '')}: " + TOKEN.sub("<T>", prompt)))
for key in sorted(k for k in examples if k not in ("hl", "hl_ref", "hlm")):
    for row, rec, sentences, loss, token, neg, shape in examples[key][:2]:
        with Image.open(rd.MERGED_ROOT / rec["source_image"]) as im:
            src = np.asarray(im.convert("RGB")).astype(float)
        with Image.open(rd.MERGED_ROOT / rec["target_image"]) as im:
            tgt = np.asarray(im.convert("RGB")).astype(float)
        a = src.copy()
        a[neg] = 0.5 * a[neg] + [0, 0, 127]
        for t in token: a[t] = 0.5 * a[t] + [0, 127, 0]
        for l in loss: a[l] = 0.5 * a[l] + [127, 0, 0]
        both = np.concatenate([a, tgt], 1).clip(0, 255).astype(np.uint8)
        text = key + ": " + " | ".join(TOKEN.sub("<T>", s) for s in sentences)
        tiles.append(tile(Image.fromarray(both), text))
cols = 4
H = (len(tiles) + cols - 1) // cols
out = Image.new("RGB", (360 * cols, 420 * H), "white")
for i, t in enumerate(tiles):
    out.paste(t, ((i % cols) * 360, (i // cols) * 420))
out.save(sheet, quality=85)
print("sheet", sheet, len(tiles))
