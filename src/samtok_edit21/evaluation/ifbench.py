"""IF-Bench (fairness-v2.2-1mp) evaluation of the full method: Pass1 localization, then token-only S2 editing.

The method reads only the benchmark's shared public requests
(``<bundle>/assets/shared_inputs/requests/<setting>/<case>/public.json``): the approximately 1MP canvas I with its gray
letterbox, and per unit the public instruction (ref or no-ref) and the public locator (none / point / box / canvas
mask).  The annotated canvas A is not used; regions enter as the method's own inputs.  Nothing here reads the
manifest, the ground truth or the canonical source image.  Every request is one joint pass per stage, seed 42,
without retries.

1. ``pass1`` (Pass1 environment, transformers 5.19): the SAMTok codec encodes every supplied mask on the canvas; a
   request with any none / point / box unit gets one greedy Pass1 generation for all its units, the mask units given
   as their input tokens (which they keep).  A unit for which Pass1 returns no single valid span stays text only:
   the pre-declared fallback, counted per unit.
2. ``pass2`` (v4 environment): each unit's public instruction with its token right after the selector phrase
   (``place_token``), all units as the numbered lines of the training-time multi-edit prompt; S2 DiT adapter with its
   code injector, raw SAMTok TE, 40 steps, token guidance 2.0 (CFG against the same prompt without tokens), generated
   at exactly the canvas size.  ``raw.png`` is the model output, ``output.png`` the benchmark's restoration of it
   (content crop, one LANCZOS resize to the canonical source size).
3. The canonical Judge runs in the frozen Judge runtime (``scripts/ifbench/judge.py``).

Each machine owns the images ``case_ids[node::nodes]`` with all their settings, as in the benchmark's four-node run;
the workers of a machine take its requests by an atomic claim.  A claim without a terminal record (a worker died) is
never re-run automatically: ``open-claims`` lists such requests and ``release`` is the logged, manual way out.

    python -m samtok_edit21.evaluation.ifbench prepare --bundle <bundle> --run <run dir> --config <method.json> \\
        [--cases v2-sa_1 ...] [--settings S1 ...] [--expect 10840]
    python -m samtok_edit21.evaluation.ifbench pass1 --run <run dir> --worker 0 --workers 8     # Pass1 environment
    python -m samtok_edit21.evaluation.ifbench pass2 --run <run dir> --worker 0 --workers 8     # v4 environment
    python -m samtok_edit21.evaluation.ifbench status --run <run dir>
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

SETTINGS = ["S1", "S2-point", "S2-box", "S2-mask", "S3-point", "S3-box", "S3-mask", "S4-ref", "S4-noref", "S5"]
SELECTED = "the selected region"
# Fixed protocol-2.0 suffixes after a remove selector; the token goes before them.
REMOVE_SUFFIXES = (", including its contents", " to make knee-length shorts", " to make it knee-length")
METHOD_KEYS = {"name", "qwen", "samtok", "pass1_adapter", "dit_adapter", "steps", "token_guidance", "seed",
               "pass1_max_new_tokens", "pass1_max_pixels"}
# Optional method keys of diagnostic runs: "lora_scale" scales the DiT LoRA; "drop_tokens": true edits from the public
# instructions alone (the Pass1 tokens are read but not written into the prompt, so no token guidance either).
OPTIONAL_KEYS = {"lora_scale", "drop_tokens"}
STAGES = {"pass1": "pass1", "pass2": "outputs"}
# A worker stops after this many failed requests in a row, so that a broken card cannot fail the whole queue.
MAX_CONSECUTIVE_FAILURES = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def create_json(path, value):
    """Write once; an existing file must hold the same value."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"{path} exists with a different value")
        return
    write_json(path, value)


def selector_end(instruction, operation, has_ref):
    """Index right after the selector phrase of a public protocol-2.0 instruction, from the public text only.

    No-ref: after "the selected region".  Ref (rule fixed on the training data, D10-D15: 97.8% of the renderer's
    placements reproduced exactly): add and remove end with the selector, before a fixed remove suffix; replace has
    it before the last " with "; attribute before the last " to " of its first clause, or before the value of
    "Make <selector> transparent|translucent|opaque" and "Tint <selector> <color>"."""
    text = instruction[:-1] if instruction.endswith(".") else instruction
    if not has_ref:
        start = text.find(SELECTED)
        if start < 0 or text.find(SELECTED, start + 1) >= 0:
            raise ValueError(f"No-ref instruction without exactly one {SELECTED!r}: {instruction!r}")
        return start + len(SELECTED)
    if operation in ("add", "remove"):
        for suffix in REMOVE_SUFFIXES:
            if text.endswith(suffix):
                return len(text) - len(suffix)
        return len(text)
    if operation == "replace":
        found = [m.start() for m in re.finditer(r" with ", text)]
        return found[-1] if found else len(text)
    if operation != "attribute":
        raise ValueError(f"Unknown operation {operation!r}")
    m = re.match(r"^(Make) (.+) (transparent|translucent|opaque)$", text)
    if m:
        return len(m.group(1)) + 1 + len(m.group(2))
    if text.startswith("Tint "):
        return text.rfind(" ")
    found = [m.start() for m in re.finditer(r" to ", text.split(" and its ")[0])]
    return found[-1] if found else len(text)


def place_token(instruction, operation, has_ref, span):
    """The public instruction with the region token after its selector phrase, as in training."""
    i = selector_end(instruction, operation, has_ref)
    return instruction[:i] + " " + span + instruction[i:]


def unit_sentences(public, tokens):
    """Pass2 sentence per unit: the public instruction, with its token when the unit has one (else text only)."""
    sentences = []
    for unit in public["units"]:
        span = tokens.get(unit["id"])
        sentences.append(place_token(unit["instruction"], unit["operation"], unit["has_ref"], span) if span
                         else unit["instruction"])
    return sentences


def pass2_prompt(public, tokens):
    from samtok_edit21.preparation.round2_data import multi_edit_prompt

    return multi_edit_prompt(unit_sentences(public, tokens))


# ---------------------------------------------------------------------------------------------- requests and run


def load_request(directory):
    """``(public, geometry, geometry_path)`` after the benchmark's own checks: the public schema, digest and asset
    bytes, the 1MP canvas protocol, and the exact canvas geometry of the case."""
    from samtok_benchmark.ifbench import canvas
    from samtok_benchmark.ifbench.public import verify

    public = json.loads((Path(directory) / "public.json").read_text())
    verify(public, assets=True)
    if public["protocol"] != canvas.PROTOCOL:
        raise ValueError(f"{directory}: not the shared 1MP canvas protocol")
    geometry_path = Path(public["source"]["path"]).parent / "geometry.json"
    geometry = json.loads(geometry_path.read_text())
    canvas.validate_geometry(geometry)
    if public["source"]["size"] != geometry["canvas_size"]:
        raise ValueError(f"{directory}: public source is not the registered canvas")
    return public, geometry, geometry_path


def prepare(bundle, run, config_path, cases=None, settings=None, expect=None):
    """Freeze a run: the method configuration and the verified list of public requests (CPU only)."""
    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    config = json.loads(Path(config_path).read_text())
    if set(config) - OPTIONAL_KEYS != METHOD_KEYS or config["seed"] != 42:
        raise ValueError(f"Method config needs exactly {sorted(METHOD_KEYS)} (optional {sorted(OPTIONAL_KEYS)}) and seed 42")
    settings = settings or SETTINGS
    if any(s not in SETTINGS for s in settings):
        raise ValueError(f"Unknown setting in {settings}")
    root = bundle / "assets/shared_inputs/requests"
    if cases is None:
        cases = sorted(p.name for p in (root / settings[0]).iterdir() if p.is_dir())

    def entry(pair):
        setting, case = pair
        directory = root / setting / case
        public, _, geometry_path = load_request(directory)
        if public["setting"] != setting or public["case_id"] != case:
            raise ValueError(f"{directory}: request identity differs")
        return {"request_id": public["request_id"], "setting": setting, "case_id": case,
                "public": str(directory / "public.json"),
                "public_sha256": sha256_file(directory / "public.json"),
                "public_digest": public["input_digest"],
                "geometry": str(geometry_path), "geometry_sha256": sha256_file(geometry_path),
                "units": len(public["units"]),
                "locators": [u["locator"] for u in public["units"]]}

    with ThreadPoolExecutor(max_workers=16) as pool:
        requests = list(pool.map(entry, [(s, c) for s in settings for c in cases]))
    if expect is not None and len(requests) != expect:
        raise ValueError(f"Expected {expect} requests, the bundle gives {len(requests)}")
    ready = bundle / "READY.json"
    plan = {"schema": "samtok-ifbench-1mp-run-v1", "bundle": str(bundle), "method": config,
            "bundle_ready_sha256": sha256_file(ready) if ready.exists() else None,
            "settings": settings, "case_ids": cases, "requests": requests}
    create_json(run / "RUN.json", plan)
    return plan


def load_run(run):
    plan = json.loads((Path(run) / "RUN.json").read_text())
    if plan["schema"] != "samtok-ifbench-1mp-run-v1":
        raise ValueError("Unknown run schema")
    return plan


def checked_request(item):
    for key in ("public", "geometry"):
        if sha256_file(item[key]) != item[key + "_sha256"]:
            raise ValueError(f"{item[key]} changed since the run was prepared")
    public, geometry, _ = load_request(Path(item["public"]).parent)
    return public, geometry


def request_dir(run, stage, item):
    return Path(run) / STAGES[stage] / item["setting"] / item["case_id"]


def request_state(run, stage, item):
    """``terminal`` (a result record exists), ``open`` (claimed, no result yet) or ``pending``."""
    directory = request_dir(run, stage, item)
    if (directory / "result.json").exists():
        return "terminal"
    return "open" if (directory / "claim").exists() else "pending"


def worker_requests(plan, node, nodes, worker, workers):
    """The requests of this machine's images; evenly spaced starts keep its workers out of each other's way."""
    mine = set(plan["case_ids"][node::nodes])
    items = [item for item in plan["requests"] if item["case_id"] in mine]
    start = worker * len(items) // workers
    return items[start:] + items[:start]


def claim(directory, item, stage):
    """Take a request atomically; False when it is finished or another worker holds it."""
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        return False
    try:
        (directory / "claim").mkdir()
    except FileExistsError:
        return False
    write_json(directory / "claim" / "owner.json", {"request_id": item["request_id"], "stage": stage,
                                                    "pid": os.getpid(), "host": socket.gethostname(),
                                                    "time": time.time()})
    return True


def open_claims(run):
    """``[(stage, request_id)]``: claimed without a terminal record.  Meaningful only while no worker is running."""
    plan = load_run(run)
    pairs = [(stage, item) for stage in STAGES for item in plan["requests"]]
    with ThreadPoolExecutor(max_workers=16) as pool:
        states = list(pool.map(lambda pair: request_state(run, *pair), pairs))
    return [(stage, item["request_id"]) for (stage, item), state in zip(pairs, states) if state == "open"]


def release(run, stage, request_id, reason):
    """Manual decision after an infrastructure failure: the request may run again (same input and seed), logged."""
    plan = load_run(run)
    item = next(i for i in plan["requests"] if i["request_id"] == request_id)
    directory = request_dir(run, stage, item)
    if (directory / "result.json").exists():
        raise ValueError(f"{request_id} has a terminal record; it is never released")
    released = f"claim.released.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    (directory / "claim").rename(directory / released)
    with open(Path(run) / "RELEASES.jsonl", "a") as f:
        f.write(json.dumps({"stage": stage, "request_id": request_id, "reason": reason, "moved_to": released,
                            "time": time.time()}, ensure_ascii=False) + "\n")


def status(run):
    """Progress of a run from its records: request states per stage, failures and text-only fallbacks."""
    plan = load_run(run)

    def one(pair):
        stage, item = pair
        state = request_state(run, stage, item)
        if state != "terminal":
            return stage, state, 0
        record = json.loads((request_dir(run, stage, item) / "result.json").read_text())
        return stage, record["status"], record.get("fallback_units") or 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(one, [(stage, item) for stage in STAGES for item in plan["requests"]]))
    result = {"requests": len(plan["requests"])}
    for stage in STAGES:
        result[stage] = dict(Counter(state for s, state, _ in rows if s == stage))
    result["fallback_units"] = sum(n for s, _, n in rows if s == "pass1")
    return result


class FailureGuard:
    """Stop a worker on a fatal CUDA state (the benchmark worker's rule) or after repeated failures; the record of the
    failed request is already saved, later requests stay unattempted."""

    def __init__(self):
        self.row = 0

    def check(self, record):
        self.row = self.row + 1 if record["status"].startswith("failed") else 0
        if record.get("fatal_cuda_error"):
            raise RuntimeError(f"Fatal CUDA state at {record['request_id']}; this worker stops")
        if self.row >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(f"{self.row} failed requests in a row, last {record['request_id']}; this worker stops")


def failure_fields(error):
    from samtok_benchmark.ifbench.fairness_dev_worker import fatal_cuda_error

    return {"status": "failed", "error": repr(error), "error_type": type(error).__name__,
            "traceback": traceback.format_exc(), "fatal_cuda_error": fatal_cuda_error(error)}


# ---------------------------------------------------------------------------------------------------- pass 1


# One requested item whose "mask_2d" key name may be missing: {"label":"U4":"<span>"} for {"label":"U4","mask_2d":"<span>"}.
LOOSE_ITEM = re.compile(r'\{"label":"(U[1-5])"(?:,"mask_2d")?:"(<\|mt_start\|><\|mt_[0-9]{4}\|><\|mt_[0-9]{4}\|><\|mt_end\|>)"\}')


def recover_missing_key(raw, labels):
    """Format repair of a Pass1 answer whose only defect is a missing "mask_2d" key name: the spans of ``labels`` in
    order, else None.  Everything else must be the plain JSON array of exactly these items; no span is changed."""
    from samtok_edit21.pass1.protocol import valid_span

    text = raw.strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    items = list(LOOSE_ITEM.finditer(text))
    if [m[1] for m in items] != list(labels) or not all(valid_span(m[2]) for m in items):
        return None
    if LOOSE_ITEM.sub("{}", text) != "[" + ",".join("{}" for _ in items) + "]":
        return None
    return {m[1]: m[2] for m in items}


def mask_iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


class Pass1Models:
    """The codec and the grounder, loaded when a request first needs them."""

    def __init__(self, method):
        self.method = method
        self._codec = self._grounder = None

    @property
    def codec(self):
        if self._codec is None:
            from samtok_edit21.models.codec import load_codec

            self._codec = load_codec(self.method["samtok"], "cuda:0")
        return self._codec

    @property
    def grounder(self):
        if self._grounder is None:
            from samtok_edit21.pass1.grounder import Pass1Grounder

            self._grounder = Pass1Grounder(adapter=self.method["pass1_adapter"], base_model=self.method["samtok"],
                                           max_pixels=self.method["pass1_max_pixels"])
        return self._grounder


def ground_request(models, method, item, out):
    """Tokens for one request: the codec for supplied masks, one joint Pass1 generation for all other units."""
    import numpy as np
    from PIL import Image

    from samtok_edit21.pass1.evaluation.ifbench import adapt
    from samtok_edit21.pass1.protocol import valid_span

    public, _ = checked_request(item)
    with Image.open(public["source"]["path"]) as im:
        image = im.convert("RGB")
    rgba = image.convert("RGBA")
    cache, units = {}, {}
    for unit in public["units"]:
        units[unit["id"]] = {"id": unit["id"], "operation": unit["operation"], "has_ref": unit["has_ref"],
                             "locator": unit["locator"], "token": None, "source": None}
        if unit["locator"] != "mask":
            continue
        with Image.open(unit["region"]["path"]) as im:
            mask = np.asarray(im.convert("L")) > 0
        span = models.codec.encode(rgba, [mask])[0][0]
        decoded = models.codec.decode_strict(rgba, span)[0] > 0
        cache[(public["source"]["sha256"], unit["region"]["sha256"])] = span
        units[unit["id"]].update(token=span, source="codec", area=float(mask.mean()),
                                 decoded_area=float(decoded.mean()), roundtrip_iou=mask_iou(decoded, mask))
    fields = {"pass1": None}        # every unit supplied its mask: no Pass1 call
    if any(u["locator"] != "mask" for u in public["units"]):
        request = adapt(public, cache)
        start = time.monotonic()
        answer = models.grounder.ground(image, request, max_new_tokens=method["pass1_max_new_tokens"])
        fields["pass1"] = {"request": request, "raw": answer["raw"], "parsed": answer["parsed"],
                           "complete": answer["parsed"]["complete"], "seconds": time.monotonic() - start}
        for unit in public["units"]:
            if unit["locator"] == "mask":
                continue            # the supplied mask keeps its input token
            span = answer["parsed"]["spans"].get(unit["id"])
            if span and valid_span(span):
                decoded = models.codec.decode_strict(rgba, span)[0] > 0
                Image.fromarray(decoded.astype(np.uint8) * 255).save(out / f"{unit['id']}.png")
                units[unit["id"]].update(token=span, source="pass1", decoded_area=float(decoded.mean()))
            else:
                units[unit["id"]].update(source="fallback_text")
    fields.update(status="tokens", units=[units[u["id"]] for u in public["units"]],
                  fallback_units=sum(u["source"] == "fallback_text" for u in units.values()))
    return fields


def run_pass1(run, node=0, nodes=1, worker=0, workers=1):
    import torch

    plan = load_run(run)
    method = plan["method"]
    models, guard = Pass1Models(method), FailureGuard()
    done = 0
    for item in worker_requests(plan, node, nodes, worker, workers):
        out = request_dir(run, "pass1", item)
        if not claim(out, item, "pass1"):
            continue
        tick = time.monotonic()
        record = {"request_id": item["request_id"], "setting": item["setting"], "case_id": item["case_id"],
                  "public_sha256": item["public_sha256"], "pass1_adapter": method["pass1_adapter"]}
        try:
            record.update(ground_request(models, method, item, out))
        except Exception as error:
            record.update(failure_fields(error))
        record["wall_seconds"] = time.monotonic() - tick
        record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        create_json(out / "result.json", record)
        done += 1
        print(json.dumps({"stage": "pass1", "node": node, "worker": worker, "done": done,
                          "request": item["request_id"], "status": record["status"],
                          "fallback": record.get("fallback_units"), "sec": round(record["wall_seconds"], 1)}),
              flush=True)
        guard.check(record)


# ---------------------------------------------------------------------------------------------------- pass 2


def generate_request(pipe, method, item, tokens, out):
    """One joint S2 generation for a request; returns the result fields of ``generated``."""
    import torch
    from PIL import Image
    from samtok_benchmark.ifbench.canvas import restore_output

    from samtok_edit21.data.protocol import strip_tokens
    from samtok_edit21.models.pipeline import edit

    public, geometry = checked_request(item)
    spans = {} if method.get("drop_tokens") else {u["id"]: u["token"] for u in tokens["units"] if u["token"]}
    prompt = pass2_prompt(public, spans)
    guided = bool(spans) and method["token_guidance"] != 1.0
    parameters = {"num_inference_steps": method["steps"], "seed": method["seed"],
                  "width": geometry["canvas_size"][0], "height": geometry["canvas_size"][1],
                  "cfg_scale": method["token_guidance"] if guided else 1.0, "use_kv_cache": True,
                  "pass2_te": "raw", "mode": "inline" if spans else "direct"}
    if guided:
        parameters["negative_prompt"] = strip_tokens(prompt)
    create_json(out / "call.json", {"request_id": item["request_id"], "images": [public["source"]],
                                    "prompt": prompt, "parameters": parameters, "dit_adapter": method["dit_adapter"],
                                    "tokens": {u["id"]: u["source"] for u in tokens["units"]}})
    with Image.open(public["source"]["path"]) as im:
        image = im.convert("RGB")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    # The compiled FlexAttention kernels depend on the canvas shapes this process has seen: tiny numeric differences that
    # diffusion amplifies (about 0.5 of 255 per pixel).  A fresh compile state makes the bytes a function of the request.
    torch._dynamo.reset()
    raw, _ = edit(pipe, prompt, [image.convert("RGBA")], **parameters)
    seconds = time.monotonic() - start
    if list(raw.size) != geometry["canvas_size"]:
        raise RuntimeError(f"Output {raw.size} is not the canvas {geometry['canvas_size']}")
    raw.save(out / "raw.png")
    output = restore_output(raw, geometry)
    output.save(out / "output.png")
    return {"status": "generated", "prompt": prompt, "parameters": parameters, "generate_seconds": seconds,
            "transformer_forward_calls": method["steps"] * (2 if guided else 1),
            "raw_image": str(out / "raw.png"), "raw_sha256": sha256_file(out / "raw.png"),
            "raw_size": list(raw.size), "raw_mode": raw.mode, "output_image": str(out / "output.png"),
            "output_sha256": sha256_file(out / "output.png"), "output_size": list(output.size),
            "fallback_units": tokens["fallback_units"]}


def run_pass2(run, node=0, nodes=1, worker=0, workers=1):
    import torch

    from samtok_edit21.models.adapters import load_grounding_adapter
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline

    plan = load_run(run)
    method = plan["method"]
    mine = worker_requests(plan, node, nodes, worker, workers)
    missing = [item["request_id"] for item in mine if request_state(run, "pass1", item) != "terminal"]
    if missing:
        raise RuntimeError(f"Pass1 has no terminal record for {missing[:10]} ({len(missing)} in all); run pass1 first")
    if not any(request_state(run, "pass2", item) == "pending" for item in mine):
        return
    tick = time.monotonic()
    pipe = load_pipeline(method["qwen"], method["samtok"], device="cuda")
    codec = load_codec(method["samtok"], "cuda")
    load_grounding_adapter(pipe, codec, method["dit_adapter"])
    if method.get("lora_scale", 1.0) != 1.0:          # quality probe: scale the DiT LoRA contribution
        from peft.tuners.tuners_utils import BaseTunerLayer
        for module in pipe.dit.modules():
            if isinstance(module, BaseTunerLayer):
                module.scale_layer(method["lora_scale"])
    pipe.eval()
    load_seconds = time.monotonic() - tick
    guard = FailureGuard()
    done = 0
    for item in mine:
        out = request_dir(run, "pass2", item)
        if not claim(out, item, "pass2"):
            continue
        tick = time.monotonic()
        tokens_path = request_dir(run, "pass1", item) / "result.json"
        tokens = json.loads(tokens_path.read_text())
        result = {"request_id": item["request_id"], "setting": item["setting"], "case_id": item["case_id"],
                  "public_sha256": item["public_sha256"], "tokens_record": str(tokens_path),
                  "tokens_record_sha256": sha256_file(tokens_path), "load_seconds": load_seconds}
        if tokens["status"] != "tokens":    # the method's own localization failed: a failure in the denominator
            result.update(status="failed-planner", error=f"Pass1 stage: {tokens.get('error')}")
        else:
            try:
                result.update(generate_request(pipe, method, item, tokens, out))
            except Exception as error:
                result.update(failure_fields(error))
                torch.cuda.empty_cache()
        result["wall_seconds"] = time.monotonic() - tick
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        create_json(out / "result.json", result)
        done += 1
        print(json.dumps({"stage": "pass2", "node": node, "worker": worker, "done": done,
                          "request": item["request_id"], "status": result["status"],
                          "sec": round(result["wall_seconds"], 1)}), flush=True)
        guard.check(result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("action", choices=["prepare", "pass1", "pass2", "status", "open-claims", "release"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--cases", nargs="*")
    p.add_argument("--settings", nargs="*")
    p.add_argument("--expect", type=int)
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stage", choices=sorted(STAGES), help="release: the stage of the request")
    p.add_argument("--request", help="release: the request id")
    p.add_argument("--reason", help="release: why this request may run again")
    a = p.parse_args(argv)
    if a.action == "prepare":
        plan = prepare(a.bundle, a.run, a.config, a.cases or None, a.settings or None, a.expect)
        print(json.dumps({"requests": len(plan["requests"]), "cases": len(plan["case_ids"]),
                          "settings": plan["settings"]}))
    elif a.action == "pass1":
        run_pass1(a.run, a.node, a.nodes, a.worker, a.workers)
    elif a.action == "pass2":
        run_pass2(a.run, a.node, a.nodes, a.worker, a.workers)
    elif a.action == "status":
        print(json.dumps(status(a.run), ensure_ascii=False, indent=1))
    elif a.action == "open-claims":
        found = open_claims(a.run)
        print(json.dumps({"open_claims": found}, ensure_ascii=False))
        return 1 if found else 0
    else:
        if not (a.stage and a.request and a.reason):
            p.error("release needs --stage, --request and --reason")
        release(a.run, a.stage, a.request, a.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
