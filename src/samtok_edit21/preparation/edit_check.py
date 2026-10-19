"""MLLM edit check of cleaned training pairs (scheme 2 of docs/archive/v2/09 section 6.11).

Qwen3.8-27B behind a vLLM OpenAI-compatible server runs two independent checks:
  q2   one thinking call answering short per-type questions;
  dn2  a blind description of the change (no thinking, instruction withheld),
       then a thinking call that compares the instruction with the description.
A pair is kept only when both checks return "pass".  Views, prompts and sampling
are the ones validated in the pilot (docs/archive/v2/09 section 6.11); any change to them
invalidates the pilot's recall / false-alarm numbers, so tests pin their hashes.
"""
from __future__ import annotations

import asyncio
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FULL_PIXELS = 640 * 640          # full BEFORE / AFTER views
CLOSE_LONG_SIDE = 1400           # side-by-side close-up of the target area
CLOSE_MAX_SCALE = 3.0
CONTOUR_COLOR = (235, 50, 45)    # first colour of the benchmark judge's outlines
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# The pilot's label font (Debian fonts-dejavu-core). Machines without that package point
# SAMTOK_CHECK_FONT at a byte-identical copy; the ARNOLD launcher checks the hash.
FONT_SHA256 = "0d977336a6d5fba34eab8e3199eb218327161b5143749f802982c2bc34df0c96"
JPEG_QUALITY = 92

# Sampling from the model card: thinking and non-thinking presets.
THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
NON_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
# call: (thinking, reasoning effort, max output tokens)
CALLS = {"q2": (True, "low", 10000), "describe": (False, None, 6000), "judge2": (True, "low", 8000)}
# Requests per call before a pair is recorded as unchecked; attempt k uses seed k (the pilot used seed 0).
VERDICT_ATTEMPTS = 3
TRANSPORT_ATTEMPTS = 6

VIEW_LABELS = ("Image 1, BEFORE (original; red outline = edit target):",
               "Image 2, AFTER (edited):",
               "Image 3, close-up of the target area, BEFORE left and AFTER right:")

QUESTIONS2 = {
    "remove": "1. In the AFTER close-up, is any part of the target still visible, even redrawn, faint, or as a ghost shape?\n"
              "2. Is anything the target was holding or supporting now left floating or cut off?",
    "add": "1. What new object appears in AFTER, and where exactly?\n"
           "2. Is it the requested kind, complete and sharp, and supported the way such an object normally is (standing, hanging, attached or held)?",
    "replace": "1. Which object(s) does the instruction refer to, and is every one of them completely gone?\n"
               "2. Is the new object clearly the requested kind, including any stated size, age, species or model?",
    "attribute": "1. Is the target still the same object (same shape, model and pose)?\n"
                 "2. Name its main color or material in BEFORE and in AFTER. Does every part the instruction names now clearly show the requested attribute, not just a darker, lighter or tinted version?",
    "other": "1. What exactly changed at the target?\n2. Does the change fully match the instruction?",
}

PROMPT2 = """You are checking one training example for an instruction-based image editing model.
The images: 1 is BEFORE with the edit target outlined in red, 2 is AFTER, 3 is a close-up of the target area (BEFORE left, AFTER right).
Edit type: {op}
Instruction: {instruction}

Look closely at the close-up and answer briefly:
{questions}
3. Is the change on the exact object or part the instruction names, and on every one of them?
4. Does every placement word in the instruction (on, in, under, held by, worn by, leaning out of, next to) hold in AFTER?
5. Did another object clearly change, or did a noticeable new object appear that the instruction does not ask for?
6. Is there anything a viewer would notice at a glance: a leftover part or ghost shape of a removed object, a blurry blob, or an object that floats or is cut off? Slight blur and soft shadows are fine.

FAIL if any answer shows a clear problem, otherwise PASS.

Return only JSON: {{"answers": "<short answers to 1-6>", "verdict": "pass" or "fail", "problem": one of "none", "not_done", "too_weak", "partial", "wrong_target", "scope", "defect", "reason": "<at most 20 words>"}}"""

DESCRIBE = """Image 1 is an original photo (BEFORE) with a red outline around one area. Image 2 is an edited version (AFTER). Image 3 is a close-up of the outlined area, BEFORE on the left and AFTER on the right.

Describe exactly what is different in AFTER, looking mainly at the close-up:
- each object that changed: what it was before and what it is now (kind, color, material, shape), and whether it is complete and sharp, or blurry, smeared, cut off or partly erased;
- for each new or replaced object: what it rests on, hangs from or is attached to; say plainly if it touches nothing and floats;
- anything left behind where an object was removed: remaining parts, an outline, a smudge, a patch or a shadow;
- any other object that changed, inside or outside the outline.
Describe only what you can see. Do not guess what the edit was meant to do. Answer in 3 to 6 short sentences."""

JUDGE2 = """You are checking one training example for an instruction-based image editing model. Image 1 is BEFORE with a red outline, image 2 is AFTER, image 3 is a close-up of the outlined area (BEFORE left, AFTER right).

Edit type: {op}
Instruction: {instruction}

Someone who did not see the instruction described the change like this:
"{description}"

Compare the instruction with the description and the images, point by point:
1. Object: is the change on the exact object or part the instruction names (for example the tongue of a shoe, not its ankle), and on every one of them if several are named?
2. Result: does the target now clearly match the request? An added or replacing object must clearly be the requested kind. A new color must clearly be the requested color, not just a darker or lighter version of the old one.
3. Identity (attribute edits only): the target must stay the same object, with the same shape, model and pose. Turning it into a different object fails.
4. Relations: every placement word in the instruction (on, in, under, held by, worn by, leaning out of, next to) must hold in AFTER.
5. Side effects: did another object clearly change, or did a noticeable new object appear that the instruction does not ask for (for example a new panel, sign or second copy)?
6. Artifacts: is there something a viewer would notice at a glance: a recognizable leftover part or ghost shape of a removed object, a blurry blob, or an object that floats or is cut off? Slight blur, soft shadows and faint texture differences where something was removed are normal and fine.

FAIL if any point clearly fails, otherwise PASS.

Return only JSON: {{"verdict": "pass" or "fail", "problem": one of "none", "not_done", "too_weak", "partial", "wrong_target", "scope", "defect", "reason": "<at most 20 words>"}}"""


def op_of(edit_type):
    """Prompt edit type: the four main types; action and text are attribute edits; the rest is "other"."""
    if edit_type in ("add", "remove", "replace", "attribute"):
        return edit_type
    return "attribute" if edit_type in ("action", "text") else "other"


def q2_prompt(row):
    op = op_of(row["type"])
    return PROMPT2.format(op=op, instruction=row["instruction"].strip(), questions=QUESTIONS2[op])


def judge2_prompt(row, description):
    return JUDGE2.format(op=op_of(row["type"]), instruction=row["instruction"].strip(),
                         description=description.replace('"', "'"))


def protocol():
    """Everything that defines the check; its hash goes into the run identity."""
    value = {"prompts": {"q2": PROMPT2, "questions2": QUESTIONS2, "describe": DESCRIBE, "judge2": JUDGE2},
             "views": {"labels": VIEW_LABELS, "full_pixels": FULL_PIXELS, "close_long_side": CLOSE_LONG_SIDE,
                       "close_max_scale": CLOSE_MAX_SCALE, "contour_color": CONTOUR_COLOR, "jpeg_quality": JPEG_QUALITY},
             "calls": CALLS, "thinking": THINKING, "non_thinking": NON_THINKING,
             "verdict_attempts": VERDICT_ATTEMPTS, "decision": "keep only if q2 and dn2 both return pass"}
    return {**value, "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}


def resized(image, max_pixels):
    image = image.convert("RGB")
    scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
    if scale >= 1:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def contoured(image, mask, max_pixels):
    """Red contour with a black halo just outside the mask (the benchmark judge's drawing)."""
    image = resized(image, max_pixels)
    mask = mask.convert("L").point(lambda x: 255 if x > 0 else 0).resize(image.size, Image.Resampling.NEAREST)
    width = max(2, round(max(image.size) / 400))
    outer = mask.filter(ImageFilter.MaxFilter(2 * (width + 1) + 1))
    inner = mask.filter(ImageFilter.MaxFilter(2 * width + 1))
    image.paste((0, 0, 0), mask=ImageChops.subtract(outer, mask))
    image.paste(CONTOUR_COLOR, mask=ImageChops.subtract(inner, mask))
    return image


def crop_box(box_1000, size, pad=0.25, min_frac=0.2):
    """Region box grown by `pad` of its size on each side, at least `min_frac` of the long image side, inside the image."""
    W, H = size
    x1, y1, x2, y2 = box_1000
    x1, x2, y1, y2 = x1 / 1000 * W, x2 / 1000 * W, y1 / 1000 * H, y2 / 1000 * H
    w, h = x2 - x1, y2 - y1
    side = max(max(w, h) * (1 + 2 * pad), min_frac * max(W, H))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = min(max(side, w * (1 + 2 * pad)), W), min(max(side, h * (1 + 2 * pad)), H)
    left = min(max(0, cx - bw / 2), W - bw)
    top = min(max(0, cy - bh / 2), H - bh)
    return round(left), round(top), round(left + bw), round(top + bh)


def font_path():
    return os.environ.get("SAMTOK_CHECK_FONT") or FONT


def labeled(image, text):
    out = Image.new("RGB", (image.width, image.height + 34), "white")
    out.paste(image, (0, 34))
    ImageDraw.Draw(out).text((8, 6), text, font=ImageFont.truetype(font_path(), 22), fill=(0, 0, 0))
    return out


def side_by_side(source, target, mask, box_1000):
    """Close-up of the target area: BEFORE with the contour on the left, AFTER on the right."""
    box = crop_box(box_1000, source.size)
    before = contoured(source.crop(box), mask.crop(box), 10 ** 9)
    after = target.convert("RGB").crop(box)
    scale = min((CLOSE_LONG_SIDE / 2 - 10) / max(before.width, before.height), CLOSE_MAX_SCALE)
    size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    before = labeled(before.resize(size, Image.Resampling.LANCZOS), "BEFORE (close-up)")
    after = labeled(after.resize(size, Image.Resampling.LANCZOS), "AFTER (close-up)")
    canvas = Image.new("RGB", (before.width * 2 + 20, before.height), "white")
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + 20, 0))
    return canvas


def views(source, target, mask, box_1000):
    return [(VIEW_LABELS[0], contoured(source, mask, FULL_PIXELS)),
            (VIEW_LABELS[1], resized(target, FULL_PIXELS)),
            (VIEW_LABELS[2], side_by_side(source, target, mask, box_1000))]


def data_url(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def region_mask(region):
    from pycocotools import mask as mask_utils
    mask = mask_utils.decode({"size": region["rle_size"], "counts": region["rle_counts"].encode()})
    return Image.fromarray(mask.astype(np.uint8) * 255)


def image_parts(row, root):
    """Chat content parts shared by the three calls of a pair (labels and JPEG images, no prompt)."""
    root = Path(root)
    source, target = Image.open(root / row["source_image"]), Image.open(root / row["target_image"])
    parts = []
    for label, image in views(source, target, region_mask(row["region"]), row["region"]["box_1000"]):
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url", "image_url": {"url": data_url(image)}})
    return parts


def _image_parts(job):
    row, root = job
    return image_parts(row, root)


def parse_verdict(text):
    raw = text.rsplit("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    if not isinstance(value, dict) or value.get("verdict") not in ("pass", "fail"):
        raise ValueError(f"no pass/fail verdict: {raw[:200]}")
    return value


class Server:
    """OpenAI-compatible chat endpoint; transport errors are retried, then raised."""

    def __init__(self, session, url, model):
        self.session, self.url, self.model = session, url, model

    async def call(self, parts, prompt, call, seed):
        thinking, effort, max_tokens = CALLS[call]
        body = {"model": self.model, "max_tokens": max_tokens, "seed": seed,
                "messages": [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}],
                "chat_template_kwargs": {"enable_thinking": thinking, **({"reasoning_effort": effort} if effort else {})},
                **(THINKING if thinking else NON_THINKING)}
        error = None
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                async with self.session.post(self.url, json=body) as response:
                    data = await response.json(content_type=None)
                if "choices" not in data:
                    raise RuntimeError(str(data)[:500])
                choice = data["choices"][0]
                return {"text": choice["message"].get("content") or "", "finish": choice.get("finish_reason"),
                        "tokens": (data.get("usage") or {}).get("completion_tokens", 0)}
            except Exception as exc:  # noqa: BLE001 - retried, then raised below
                error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 + 5 * attempt)
        raise RuntimeError(f"{call} request failed {TRANSPORT_ATTEMPTS} times: {error}")


async def verdict(server, parts, prompt, call, extra=None):
    """A pass/fail call, retried with new seeds while the answer has no verdict."""
    tokens, finishes = 0, []
    for attempt in range(VERDICT_ATTEMPTS):
        response = await server.call(parts, prompt, call, seed=attempt)
        tokens += response["tokens"]
        finishes.append(response["finish"])
        try:
            value = parse_verdict(response["text"])
        except Exception:  # noqa: BLE001 - malformed or truncated answer: next seed
            continue
        result = {"verdict": value["verdict"], "problem": value.get("problem"), "reason": value.get("reason")}
        if "answers" in value:
            result["answers"] = value["answers"]
        return {**result, **(extra or {}), "attempts": attempt + 1, "tokens": tokens}
    return {"verdict": None, **(extra or {}), "attempts": VERDICT_ATTEMPTS, "tokens": tokens, "finish": finishes}


async def described_verdict(server, row, parts):
    tokens = 0
    for attempt in range(VERDICT_ATTEMPTS):
        response = await server.call(parts, DESCRIBE, "describe", seed=attempt)
        tokens += response["tokens"]
        description = response["text"].rsplit("</think>", 1)[-1].strip()
        if description and response["finish"] == "stop":
            result = await verdict(server, parts, judge2_prompt(row, description), "judge2",
                                   {"description": description, "describe_attempts": attempt + 1})
            return {**result, "tokens": result["tokens"] + tokens}
    return {"verdict": None, "description": None, "describe_attempts": VERDICT_ATTEMPTS, "attempts": 0, "tokens": tokens}


async def check_pair(server, row, parts):
    start = time.monotonic()
    q2, dn2 = await asyncio.gather(verdict(server, parts, q2_prompt(row), "q2"), described_verdict(server, row, parts))
    return {"id": row["id"], "dataset": row["dataset"], "type": row["type"],
            "keep": q2["verdict"] == "pass" and dn2["verdict"] == "pass",
            "q2": q2, "dn2": dn2, "seconds": round(time.monotonic() - start, 1)}


def completed_records(path):
    """Records of an append-only result file; a torn last line (crash) is dropped and the file rewritten."""
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text()
    records, lines = {}, text.splitlines()
    for number, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise ValueError(f"Corrupt record in the middle of {path}: line {number + 1}")
            temporary = path.with_suffix(".repair.tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values()))
            temporary.replace(path)
            break
        if value["id"] in records:
            raise ValueError(f"Duplicate record {value['id']} in {path}")
        records[value["id"]] = value
    else:
        if text and not text.endswith("\n"):  # complete last record without its newline
            with path.open("a") as stream:
                stream.write("\n")
    return records


async def check_rows(rows, root, output, url, model, pairs_in_flight=256, workers=16, report=None,
                     report_every=30, stall_seconds=1800):
    """Check `rows` against the server and append one record per pair to `output`; resumes from it."""
    import aiohttp

    done = completed_records(output)
    todo = [row for row in rows if row["id"] not in done]
    counts = Counter(kept=sum(r["keep"] for r in done.values()), done=len(done))
    counts["unchecked"] = sum(r["q2"]["verdict"] is None or r["dn2"]["verdict"] is None for r in done.values())
    start, resumed = time.monotonic(), len(done)
    last = [start]
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(pairs_in_flight)
    tasks, errors = set(), []
    connector = aiohttp.TCPConnector(limit=0)  # the pilot client was capped at aiohttp's default of 100 connections
    timeout = aiohttp.ClientTimeout(total=3600)
    # spawn: never fork the event-loop process (aiohttp may hold resolver threads).
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    with pool, open(output, "a") as stream:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            server = Server(session, url, model)

            async def one(row):
                try:
                    parts = await loop.run_in_executor(pool, _image_parts, (row, str(root)))
                    record = await check_pair(server, row, parts)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    counts["done"] += 1
                    counts["kept"] += record["keep"]
                    counts["unchecked"] += record["q2"]["verdict"] is None or record["dn2"]["verdict"] is None
                    last[0] = time.monotonic()
                except Exception as exc:  # noqa: BLE001 - re-raised by the feeding loop
                    errors.append(RuntimeError(f"{row['id']}: {type(exc).__name__}: {exc}"))
                finally:
                    slots.release()

            def status():
                elapsed = time.monotonic() - start
                rate = (counts["done"] - resumed) / max(elapsed, 1e-6)
                return {"done": counts["done"], "total": len(rows), "kept": counts["kept"],
                        "unchecked": counts["unchecked"], "checked_this_run": counts["done"] - resumed,
                        "pairs_per_second": round(rate, 3), "elapsed_hours": round(elapsed / 3600, 3),
                        "eta_hours": round((len(rows) - counts["done"]) / rate / 3600, 2) if rate else None}

            async def progress():
                while True:
                    await asyncio.sleep(report_every)
                    if report:
                        try:
                            report(status())
                        except Exception as exc:  # noqa: BLE001 - progress is advisory
                            print(f"progress report failed: {type(exc).__name__}: {exc}", flush=True)
                    if tasks and time.monotonic() - last[0] > stall_seconds:
                        # A hung server would otherwise hold every slot until the request timeout.
                        errors.append(TimeoutError(f"No pair finished for {stall_seconds} s; the server looks hung"))
                        for task in list(tasks):
                            task.cancel()
                        return

            reporter = asyncio.create_task(progress())
            try:
                for row in todo:
                    await slots.acquire()
                    if errors:
                        raise errors[0]
                    task = asyncio.create_task(one(row))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                await asyncio.gather(*list(tasks), return_exceptions=True)
                if errors:
                    raise errors[0]
            finally:
                reporter.cancel()
                for task in list(tasks):
                    task.cancel()
                stream.flush()
                os.fsync(stream.fileno())
    summary = status()
    summary.pop("eta_hours")
    if report:
        report(summary)
    return summary
