from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    SamtokEditingDataset,
    make_labels,
    parse_and_canonicalize_mt_cot,
    sanitize_label,
    span_of,
    to_cot,
)
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder_samtok import (  # noqa: E402
    QwenImageSamtokTextEncoderStateDictConverter,
)
from samtok_codec import SamtokCodec  # noqa: E402


SPAN_A = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"


class SamtokEditTests(unittest.TestCase):
    def test_canonical_cot_and_label_rules(self):
        self.assertEqual(span_of([1, 257]), SPAN_A)
        self.assertEqual(make_labels("left cat", 1), ["left cat"])
        self.assertEqual(make_labels("two cats", 2), ["one of the two cats"] * 2)
        self.assertEqual(sanitize_label('  bad\n"label" `x` \\ '), "bad 'label' 'x' '")
        self.assertEqual(to_cot([]), "```json\n[]\n```")
        self.assertEqual(
            to_cot([(SPAN_A, "left cat")]),
            "```json\n"
            '[{"mask_2d": "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>", '
            '"label": "left cat"}]\n'
            "```",
        )
        with self.assertRaises(ValueError):
            to_cot(
                [("<|mt_start|><|mt_0256|><|mt_0257|><|mt_end|>", "bad")]
            )

    def test_layered_parser(self):
        cases = [
            ("```json\n[]\n```", "empty", to_cot([])),
            ("No target.", "empty", to_cot([])),
            (
                json.dumps([{"mask_2d": SPAN_A, "label": "cat"}]),
                "strict",
                to_cot([(SPAN_A, "cat")]),
            ),
            (
                f'broken {{"mask_2d": {SPAN_A}, "label": "cat"}} tail',
                "item",
                to_cot([(SPAN_A, "cat")]),
            ),
            (
                f"junk {SPAN_A} and {SPAN_A}",
                "span",
                to_cot([(SPAN_A, "target")]),
            ),
            ("nothing useful", "invalid", None),
        ]
        for text, layer, expected in cases:
            with self.subTest(layer=layer, text=text):
                actual, actual_layer = parse_and_canonicalize_mt_cot(
                    text, return_layer=True
                )
                self.assertEqual(actual_layer, layer)
                self.assertEqual(actual, expected)

    def test_schedule_is_rank_homogeneous_and_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for sample_type, count in [
                ("edit_mt", 5),
                ("edit_ntp", 3),
                ("edit", 4),
            ]:
                for index in range(count):
                    row = {"prompt": str(index), "sample_type": sample_type}
                    if sample_type != "edit":
                        row["mt_cot"] = to_cot([(SPAN_A, "target")])
                    rows.append(row)
            metadata = folder / "metadata.jsonl"
            metadata.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = SamtokEditingDataset(
                base_path=str(folder),
                metadata_path=str(metadata),
                repeat=2,
                data_file_keys=[],
                type_ratio="edit_mt:2,edit_ntp:1,edit:1",
                num_processes=2,
                gradient_accumulation_steps=4,
                seed=3,
            )
            scheduled = [
                dataset.data[index]["sample_type"] for index in dataset.schedule
            ]
            for offset in range(0, len(scheduled), 8):
                step = scheduled[offset : offset + 8]
                self.assertEqual(Counter(step), Counter(edit_mt=4, edit_ntp=2, edit=2))
                self.assertTrue(
                    all(step[substep] == step[substep + 1] for substep in range(0, 8, 2))
                )
            self.assertEqual(len(dataset) % 8, 0)

    def test_schedule_rejects_noncanonical_cot(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata = folder / "bad.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "prompt": "x",
                        "sample_type": "edit_ntp",
                        "mt_cot": json.dumps(
                            [{"mask_2d": SPAN_A, "label": "x"}]
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                SamtokEditingDataset(
                    base_path=str(folder),
                    metadata_path=str(metadata),
                    data_file_keys=[],
                    type_ratio="edit_ntp:1",
                    gradient_accumulation_steps=1,
                )

    def test_state_dict_converter_supports_both_hf_layouts(self):
        class DiskLike:
            def __init__(self):
                self.values = {
                    "visual.patch_embed.weight": 1,
                    "model.layers.0.weight": 2,
                    "model.language_model.layers.1.weight": 3,
                    "model.visual.block.weight": 4,
                    "model.language_model.embed_tokens.weight": 5,
                }

            def __iter__(self):
                return iter(self.values)

            def __getitem__(self, key):
                return self.values[key]

        converted = QwenImageSamtokTextEncoderStateDictConverter(DiskLike())
        self.assertEqual(converted["model.visual.patch_embed.weight"], 1)
        self.assertEqual(converted["model.language_model.layers.0.weight"], 2)
        self.assertEqual(converted["model.language_model.layers.1.weight"], 3)
        self.assertEqual(converted["model.visual.block.weight"], 4)
        self.assertEqual(converted["lm_head.weight"], 5)

    def test_codec_rejects_empty_mask_list_cleanly(self):
        with self.assertRaisesRegex(ValueError, "one or more non-empty"):
            SamtokCodec._ordered_masks([])


if __name__ == "__main__":
    unittest.main()
