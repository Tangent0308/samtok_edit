"""One autoregressive generation per source image; no SAM or oracle selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

from samtok_edit21.pass1.model import encode, load_model, processor_for
from samtok_edit21.pass1.protocol import parse_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=2097152)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    p = Path(args.data)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    proc = processor_for(max_pixels=args.max_pixels)
    model = load_model("cuda:0", adapter=args.adapter)
    with out.open("w") as f, torch.inference_mode():
        for row in rows[args.shard :: args.shards]:
            with Image.open(p.parent / row["image"]) as im:
                x = encode(proc, im, row["request"])
            x = {k: v.to("cuda:0") for k, v in x.items()}
            n = x["input_ids"].shape[1]
            model.model.rope_deltas = None
            start = time.time()
            y = model.generate(
                **x, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
            )
            torch.cuda.synchronize()
            ids = y[0, n:].tolist()
            raw = proc.tokenizer.decode(ids, skip_special_tokens=False)
            result = {
                "id": row["id"],
                "group": row["group"],
                "request_sha256": hashlib.sha256(
                    json.dumps(row["request"], sort_keys=True).encode()
                ).hexdigest(),
                "raw": raw,
                "generated_ids": ids,
                "new_tokens": len(ids),
                "input_tokens": n,
                "elapsed_s": time.time() - start,
                "strict": parse_answer(raw, row["request"]),
                "compatible": parse_answer(raw, row["request"], compatible=True),
                "adapter": args.adapter,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ["id", "new_tokens", "input_tokens", "elapsed_s"]
                    }
                    | {"strict": result["strict"]["complete"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
