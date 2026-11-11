"""Semantic audit of self-distilled v2 training pairs with a local VLM (Qwen3.8-27B served by vLLM).

The deterministic gates of finalize_v2.py only establish that something changed inside the region, nothing elsewhere,
without a seam or a flat recolour.  This audit checks what they cannot see, with the same four criteria as the data
line's audit (operation correct, target achieved, non-target preserved, natural) plus the target instance:
source + composite + the record's instruction -> JSON booleans; a pair passes only if all are true.

usage: audit_v2.py --final <v2_final dir> --batches <dir> ... --port 39621 [--workers 16] [--limit N]
(the server is started separately; see scripts/audit_server.sh).  Incremental: audited ids are skipped.
"""
import argparse, base64, io, json, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request

from PIL import Image

ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")
OPS = {"attribute": "change an attribute (colour, material, texture or pattern) of the referred object",
       "replace": "replace the referred object with a different object",
       "add": "add a new object at the described place", "remove": "remove the referred object"}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"operation_correct": {"type": "boolean"}, "target_correct": {"type": "boolean"},
                         "request_fulfilled": {"type": "boolean"}, "others_preserved": {"type": "boolean"},
                         "natural": {"type": "boolean"}, "defects": {"type": "array", "items": {"type": "string"}},
                         "decision": {"enum": ["pass", "fail"]}},
          "required": ["operation_correct", "target_correct", "request_fulfilled", "others_preserved", "natural",
                       "defects", "decision"]}
PROMPT = """You are auditing one training pair for instruction-based image editing. Image 1 is the source photo; image 2 is the edited result.
Instruction: "{instruction}"
Requested operation: {op}.
Judge only what is visible and answer with a JSON object:
- operation_correct: the change in image 2 is of the requested type.
- target_correct: the change is on the object the instruction refers to, not on another object or another instance.
- request_fulfilled: the requested result is fully achieved (the requested colour/material/object is clearly present; for removal the object is gone and the area is plausibly filled).
- others_preserved: everything the instruction does not mention is unchanged.
- natural: the edited area looks like a real photograph of this scene: consistent lighting, shading and texture; no flat paint-like colour, smears, halos, visible seams, duplicated or distorted parts.
- defects: short list of visible problems (empty if none).
- decision: "pass" only if all five booleans are true, otherwise "fail"."""


def data_url(path, max_edge=1024):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(port, source, target, instruction, op):
    body = {"model": "Qwen3.8-27B", "temperature": 0, "max_tokens": 600,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "audit", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(source)}},
                {"type": "image_url", "image_url": {"url": data_url(target)}},
                {"type": "text", "text": PROMPT.format(instruction=instruction, op=OPS[op])}]}]}
    req = request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    out = json.loads(text)
    out["pass"] = out.get("decision") == "pass" and all(out.get(k) is True for k in
                                                          ("operation_correct", "target_correct", "request_fulfilled", "others_preserved", "natural"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=39621)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    final = Path(a.final); apath = final / "audit.jsonl"
    try:                                    # the server may be paused (GPUs lent out): skip this pass, retry later
        request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10).read()
    except Exception as e:  # noqa: BLE001
        print(f"audit server on port {a.port} not reachable ({e!r}); skipping this pass", flush=True)
        return
    done = set()                            # items whose request errored are retried on the next pass
    if apath.exists():
        for l in open(apath):
            r = json.loads(l)
            if "error" not in r:
                done.add(r["id"])
    specs = {}
    for b in map(Path, a.batches):
        for l in open(b / "specs.jsonl"):
            s = json.loads(l); specs[s["id"]] = s
    todo = [json.loads(l) for l in open(final / "metrics.jsonl")]
    import zlib
    todo = [r for r in todo if r.get("accepted") and r["id"] not in done and r["id"] in specs
            and zlib.crc32(r["id"].encode()) % a.shards == a.shard]
    if a.limit:
        todo = todo[: a.limit]
    print(f"to audit: {len(todo)} (already {len(done)})", flush=True)
    lock = threading.Lock(); counts = {"n": 0, "pass": 0, "error": 0}

    def work(r):
        s = specs[r["id"]]
        try:
            res = ask(a.port, ROOT / s["source_image"], r["target"], s["prompt"], s["op"])
            rec = {"id": r["id"], "op": s["op"], "family": s["family"], **res}
        except Exception as e:  # noqa: BLE001
            rec = {"id": r["id"], "op": s["op"], "family": s["family"], "pass": False, "error": repr(e)[:300]}
        with lock:
            with open(apath, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["n"] += 1; counts["pass"] += bool(rec.get("pass")); counts["error"] += "error" in rec
            if counts["n"] % 200 == 0:
                print(f"{counts['n']}/{len(todo)} pass {counts['pass']} errors {counts['error']}", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    print(f"done: {counts}", flush=True)


if __name__ == "__main__":
    main()
