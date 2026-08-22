"""SAMTok editing data utilities.

This module is the single serialization boundary for mask-token CoT text.  It
also provides the ordered Stage-1 dataset used to keep every DDP rank on the
same sample type while realizing an exact edit_mt:edit_ntp:edit ratio inside
each optimizer step.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Sequence

from .unified_dataset import UnifiedDataset


MT_START = "<|mt_start|>"
MT_END = "<|mt_end|>"
MT_FMT = "<|mt_{:04d}|>"
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
ITEM_SEP = ",\n"
FALLBACK_LABEL = "target"
MAX_LABEL_CHARS = 80
MAX_SALVAGE_ITEMS = 16

SPAN_RE = re.compile(
    r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>"
)
ITEM_RE = re.compile(
    r'\{\s*"mask_2d"\s*:\s*"?(<\|mt_start\|><\|mt_(\d{4})\|>'
    r'<\|mt_(\d{4})\|><\|mt_end\|>)"?\s*,\s*"label"\s*:\s*'
    r'"((?:[^"\\]|\\.)*)"\s*\}'
)
_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.S | re.I)
_NO_TARGET_RE = re.compile(r"\bno\s+targets?\b", re.I)


def span_of(codes: Sequence[int]) -> str:
    """Serialize the two *already offset* codebook indices as one mask span."""

    if len(codes) != CODEBOOK_DEPTH:
        raise ValueError(f"Expected {CODEBOOK_DEPTH} codes, got {len(codes)}")
    return MT_START + "".join(MT_FMT.format(int(code)) for code in codes) + MT_END


def valid_span_codes(c0: int, c1: int) -> bool:
    """Return whether codes follow the 256x2 non-shared-codebook convention."""

    return (
        0 <= c0 < CODEBOOK_SIZE
        and CODEBOOK_SIZE <= c1 < CODEBOOK_SIZE * CODEBOOK_DEPTH
    )


def is_valid_span(span: str) -> bool:
    match = SPAN_RE.fullmatch(str(span))
    return bool(match) and valid_span_codes(int(match.group(1)), int(match.group(2)))


def sanitize_label(label: object) -> str:
    """Enforce the canonical, escape-free label alphabet used for training."""

    text = "".join(ch if ch.isprintable() else " " for ch in str(label))
    text = text.replace('"', "'").replace("\\", "'").replace("`", "'")
    text = " ".join(text.split())[:MAX_LABEL_CHARS].strip()
    return text or FALLBACK_LABEL


def make_labels(expression: str, count: int) -> list[str]:
    """Build the canonical labels for one referring-expression group."""

    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    expression = sanitize_label(expression)
    if count == 0:
        return []
    return [expression] if count == 1 else [f"one of the {expression}"] * count


def to_cot(items: Iterable[tuple[str, object]]) -> str:
    """Serialize ``(mask_span, label)`` pairs to the one canonical CoT form."""

    items = list(items)
    if not items:
        return "```json\n[]\n```"
    invalid = [span for span, _ in items if not is_valid_span(span)]
    if invalid:
        raise ValueError(f"Invalid SAMTok mask span(s): {invalid[:3]}")
    body = ITEM_SEP.join(
        json.dumps(
            {"mask_2d": span, "label": sanitize_label(label)},
            ensure_ascii=False,
        )
        for span, label in items
    )
    return "```json\n[" + body + "]\n```"


def _deduplicate_items(items: Iterable[tuple[str, object]]) -> list[tuple[str, object]]:
    """Drop duplicate spans while retaining the first label and original order."""

    output: list[tuple[str, object]] = []
    seen: set[str] = set()
    for span, label in items:
        if span not in seen:
            seen.add(span)
            output.append((span, label))
    return output


def parse_and_canonicalize_mt_cot(text: object, return_layer: bool = False):
    """Recover pass-1 output using strict JSON, item, then span-level parsing.

    Recovery only discards information: invalid/out-of-codebook spans and
    duplicates are removed, but missing codes are never invented.  The return
    value is a canonical CoT string, or ``None`` for an invalid generation.
    """

    text = "" if text is None else str(text)

    def result(cot, layer):
        return (cot, layer) if return_layer else cot

    fenced = _FENCE_RE.search(text)
    body = fenced.group(1) if fenced else text
    if re.fullmatch(r"\s*\[\s*\]\s*", body):
        return result(to_cot([]), "empty")

    try:
        parsed = json.loads(body)
        if isinstance(parsed, list):
            if not parsed:
                return result(to_cot([]), "empty")
            items = _deduplicate_items(
                (item["mask_2d"], item.get("label", ""))
                for item in parsed
                if isinstance(item, dict)
                and isinstance(item.get("mask_2d"), str)
                and is_valid_span(item["mask_2d"])
            )
            if items:
                return result(to_cot(items), "strict")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass

    items = []
    for span, c0, c1, label in ITEM_RE.findall(text):
        if not valid_span_codes(int(c0), int(c1)):
            continue
        try:
            label = json.loads('"' + label + '"')
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        items.append((span, label))
    items = _deduplicate_items(items)
    if items:
        return result(to_cot(items), "item")

    spans: list[str] = []
    seen: set[str] = set()
    for c0, c1 in SPAN_RE.findall(text):
        c0_int, c1_int = int(c0), int(c1)
        if not valid_span_codes(c0_int, c1_int):
            continue
        span = span_of([c0_int, c1_int])
        if span not in seen:
            seen.add(span)
            spans.append(span)
    if spans:
        return result(
            to_cot((span, FALLBACK_LABEL) for span in spans[:MAX_SALVAGE_ITEMS]),
            "span",
        )

    if _NO_TARGET_RE.search(text):
        return result(to_cot([]), "empty")
    return result(None, "invalid")


SAMPLE_TYPES = ("edit_mt", "edit_ntp", "edit")


class SamtokEditingDataset(UnifiedDataset):
    """UnifiedDataset with an exact, DDP-aware Stage-1 sample schedule."""

    def __init__(
        self,
        *args,
        type_ratio: str | None = "edit_mt:2,edit_ntp:1,edit:1",
        num_processes: int = 1,
        gradient_accumulation_steps: int = 1,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule: list[int] | None = None
        if self.load_from_cache or type_ratio is None or type_ratio.strip().lower() in {"", "none"}:
            return

        by_type = {sample_type: [] for sample_type in SAMPLE_TYPES}
        for row_id, row in enumerate(self.data):
            sample_type = row.get("sample_type", "edit")
            if sample_type not in SAMPLE_TYPES:
                raise ValueError(f"Bad sample_type {sample_type!r} at metadata row {row_id}")
            if sample_type in {"edit_mt", "edit_ntp"}:
                cot = row.get("mt_cot")
                canonical, layer = parse_and_canonicalize_mt_cot(cot, return_layer=True)
                if not isinstance(cot, str) or canonical != cot or layer not in {"strict", "empty"}:
                    raise ValueError(
                        f"Row {row_id} ({sample_type}) must contain canonical mt_cot; "
                        "use to_cot([]) for no-target/global edits"
                    )
            by_type[sample_type].append(row_id)

        ratio: dict[str, int] = {}
        for part in type_ratio.split(","):
            try:
                name, value = part.split(":", 1)
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample type ratio component: {part!r}") from exc
            name = name.strip()
            if name not in SAMPLE_TYPES or value < 0:
                raise ValueError(f"Invalid sample type ratio component: {part!r}")
            ratio[name] = value

        active = [sample_type for sample_type in SAMPLE_TYPES if ratio.get(sample_type, 0) > 0]
        if not active:
            raise ValueError(f"No positive sample ratio in {type_ratio!r}")
        missing = [sample_type for sample_type in active if not by_type[sample_type]]
        if missing:
            raise ValueError(f"Ratio requests absent sample type(s): {missing}")

        block = [sample_type for sample_type in active for _ in range(ratio[sample_type])]
        block_size = len(block)
        process_count = int(num_processes)
        accumulation = int(gradient_accumulation_steps)
        if process_count < 1 or accumulation < 1:
            raise ValueError("num_processes and gradient_accumulation_steps must be positive")
        if accumulation % block_size:
            raise ValueError(
                f"gradient_accumulation_steps={accumulation} must be a multiple of "
                f"ratio block length {block_size}; process count {process_count} is unrestricted"
            )

        per_step = {
            sample_type: ratio[sample_type] * (accumulation // block_size) * process_count
            for sample_type in active
        }
        steps_per_repeat = max(
            math.ceil(len(by_type[sample_type]) / per_step[sample_type])
            for sample_type in active
        )
        rng = random.Random(seed)
        pools = {sample_type: [] for sample_type in active}

        def draw(sample_type: str) -> int:
            if not pools[sample_type]:
                pools[sample_type] = by_type[sample_type][:]
                rng.shuffle(pools[sample_type])
            return pools[sample_type].pop()

        schedule: list[int] = []
        for _ in range(self.repeat):
            for _ in range(steps_per_repeat):
                step_types = block * (accumulation // block_size)
                rng.shuffle(step_types)
                for sample_type in step_types:
                    schedule.extend(draw(sample_type) for _ in range(process_count))
        self.schedule = schedule
        sizes = {sample_type: len(by_type[sample_type]) for sample_type in SAMPLE_TYPES}
        print(
            "[SamtokEditingDataset] "
            f"sizes={sizes}, ratio={type_ratio}, per_step={per_step}, "
            f"steps={steps_per_repeat * self.repeat}, schedule_len={len(schedule)}"
        )

    def __len__(self):
        return super().__len__() if self.schedule is None else len(self.schedule)

    def __getitem__(self, data_id):
        if self.schedule is None:
            return super().__getitem__(data_id)
        row = self.data[self.schedule[data_id]].copy()
        for key in self.data_file_keys:
            if key in row:
                operator = self.special_operator_map.get(key, self.main_data_operator)
                row[key] = operator(row[key])
        row.setdefault("mt_cot", None)
        row.setdefault("sample_type", "edit")
        return row
