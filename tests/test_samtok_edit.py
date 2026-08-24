from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import tomllib
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    SamtokEditingDataset,
    make_labels,
    parse_and_canonicalize_mt_cot,
    sanitize_label,
    span_of,
    to_cot,
)
from diffsynth.core.data.unified_dataset import UnifiedDataset  # noqa: E402
from diffsynth.models.qwen_image_dit import QwenImageTransformerBlock  # noqa: E402
from diffsynth.diffusion import loss as loss_module  # noqa: E402
from diffsynth.pipelines.qwen_image_samtok import shifted_cot_supervision  # noqa: E402
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder_samtok import (  # noqa: E402
    QwenImageSamtokTextEncoderStateDictConverter,
)
from samtok_codec import SamtokCodec  # noqa: E402
from build_edit_ntp_metadata import (  # noqa: E402
    EDIT_VERB_TEMPLATES,
    GLOBAL_TEMPLATES,
    select_source_rows,
)
from build_edit_mt_metadata import (  # noqa: E402
    load_excluded_source_ids,
    partition_pairs,
    sample_candidates,
    source_identity_from_row,
)
from build_edit_metadata import select_candidates as select_edit_candidates  # noqa: E402
from compose_training_metadata import arrange_stage2_rows  # noqa: E402
from train_samtok_edit import (  # noqa: E402
    QwenImageSamtokTrainingModule,
    samtok_parser,
    validate_wandb_credentials,
)
from run_eval import (  # noqa: E402
    SETTING_SPECS,
    load_and_validate_rows,
    parse_settings,
    run_samtok_setting,
    stock_edit,
)
from make_stage1_category_comparisons import (  # noqa: E402
    crop_decoded_mask_cells,
    validate_matching_online_records,
)


SPAN_A = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"


class SamtokEditTests(unittest.TestCase):
    def test_category_comparison_requires_identical_stage1_stage2_online_cot(self):
        stage1 = [
            {
                "metadata_index": 0,
                "source": "source.jpg",
                "target": "target.jpg",
                "prompt": "edit it",
                "gt_mt_cot": "[]",
                "conditioned_mt_cot": "[]",
                "pass1_raw": "[]<|im_end|>",
                "parse_layer": "empty",
            }
        ]
        stage2 = [dict(stage1[0])]
        audit = validate_matching_online_records(stage1, stage2)
        self.assertTrue(audit["all_equal"])
        self.assertEqual(audit["matching_records"], 1)

        stage2[0]["conditioned_mt_cot"] = "different"
        with self.assertRaisesRegex(ValueError, "conditioned_mt_cot"):
            validate_matching_online_records(stage1, stage2)

    def test_category_mask_sheet_uses_separate_decoded_cells(self):
        panel = Image.new("RGB", (1600, 410), "white")
        for column, color in enumerate(
            [(10, 10, 10), (240, 20, 20), (20, 240, 20), (20, 20, 240), (80, 80, 80)]
        ):
            panel.paste(
                Image.new("RGB", (320, 320), color),
                (column * 320, 90),
            )

        gt, online = crop_decoded_mask_cells(panel)
        self.assertEqual(gt.size, (320, 320))
        self.assertEqual(online.size, (320, 320))
        self.assertEqual(gt.getpixel((10, 10)), (20, 240, 20))
        self.assertEqual(online.getpixel((10, 10)), (240, 20, 20))

    def test_uv_environment_definition_has_no_lockfile_workflow(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]

        self.assertEqual(project["requires-python"], "==3.11.*")
        dependencies = set(project["dependencies"])
        for required in {
            "byted-wandb==0.13.98",
            "diffsynth==2.1.2",
            "setuptools==66.1.1",
            "torch==2.8.0",
            "torchvision==0.23.0",
            "transformers==5.12.1",
        }:
            self.assertIn(required, dependencies)

        setup_script = REPO_ROOT / "setup_env.sh"
        subprocess.run(["bash", "-n", setup_script], check=True)
        script_text = setup_script.read_text(encoding="utf-8")
        self.assertIn('"$UV_EXECUTABLE" pip install', script_text)
        self.assertNotIn('"$UV_EXECUTABLE" sync', script_text)
        self.assertIn('--editable "$DIFFSYNTH_DIR"', script_text)

    def test_cache_discovery_recurses_and_selects_only_pth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "0" / "nested").mkdir(parents=True)
            (root / "1").mkdir()
            expected = {
                root / "0" / "0.pth",
                root / "0" / "nested" / "1.pth",
                root / "1" / "2.pth",
            }
            for path in expected:
                path.touch()
            (root / "0" / "0.json").touch()
            (root / "1" / "ignore.txt").touch()

            with mock.patch.dict(os.environ, {"RANK": "1"}):
                dataset = UnifiedDataset(base_path=folder, metadata_path=None)

            self.assertEqual({Path(path) for path in dataset.cached_data}, expected)

    def test_plain_edit_selection_minimizes_deprioritized_overlap(self):
        preferred = [
            ("a.parquet", index, "add", False) for index in range(3)
        ]
        overlap = [
            ("b.parquet", index, "replace", True) for index in range(5)
        ]
        selected = select_edit_candidates(preferred + overlap, 5, seed=7)
        self.assertEqual(sum(not row[3] for row in selected), 3)
        self.assertEqual(sum(row[3] for row in selected), 2)
        self.assertEqual(
            select_edit_candidates(preferred + overlap, 5, seed=7), selected
        )

    def test_stage2_composer_balances_every_strided_gpu_shard(self):
        edit_mt = [
            {"id": f"mt-{index}", "sample_type": "edit_mt"}
            for index in range(16)
        ]
        edit = [
            {"id": f"edit-{index}", "sample_type": "edit"}
            for index in range(8)
        ]
        rows, shard_counts, padding = arrange_stage2_rows(
            edit_mt, edit, random.Random(8), num_shards=8
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            shard_counts, [{"edit_mt": 2, "edit": 1} for _ in range(8)]
        )
        self.assertEqual(
            {row["id"] for row in rows},
            {row["id"] for row in edit_mt + edit},
        )
        self.assertEqual(padding, {})

    def test_stage2_composer_minimally_pads_full_2_to_1_data_for_eight_gpus(self):
        edit_mt = [
            {"id": f"mt-{index}", "sample_type": "edit_mt"}
            for index in range(110652)
        ]
        edit = [
            {"id": f"edit-{index}", "sample_type": "edit"}
            for index in range(55326)
        ]
        rows, shard_counts, padding = arrange_stage2_rows(
            edit_mt,
            edit,
            random.Random(42),
            num_shards=8,
            pad_to_shards=True,
        )
        self.assertEqual(padding, {"edit_mt": 4, "edit": 2})
        self.assertEqual(len(rows), 165984)
        self.assertEqual(
            shard_counts, [{"edit_mt": 13832, "edit": 6916} for _ in range(8)]
        )
        self.assertEqual(
            Counter(row["sample_type"] for row in rows if "schedule_padding" in row),
            {"edit_mt": 4, "edit": 2},
        )

    def test_stage2_composer_preserves_odd_full_mt_pool_with_minimal_padding(self):
        edit_mt = [
            {"id": f"mt-{index}", "sample_type": "edit_mt"}
            for index in range(110631)
        ]
        edit = [
            {"id": f"edit-{index}", "sample_type": "edit"}
            for index in range(55316)
        ]
        rows, shard_counts, padding = arrange_stage2_rows(
            edit_mt,
            edit,
            random.Random(0),
            num_shards=8,
            pad_to_shards=True,
        )
        self.assertEqual(padding, {"edit_mt": 9, "edit": 4})
        self.assertEqual(len(rows), 165960)
        self.assertEqual(
            shard_counts, [{"edit_mt": 13830, "edit": 6915} for _ in range(8)]
        )
        self.assertEqual(
            Counter(row["sample_type"] for row in rows if "schedule_padding" in row),
            {"edit_mt": 9, "edit": 4},
        )

    def test_stage2_ratio_bypass_retains_runtime_row_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata = folder / "stage2.jsonl"
            rows = [
                {"prompt": "masked", "sample_type": "edit_mt", "mt_cot": to_cot([])},
                {"prompt": "plain", "sample_type": "edit"},
            ]
            metadata.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = SamtokEditingDataset(
                base_path=str(folder),
                metadata_path=str(metadata),
                repeat=1,
                data_file_keys=[],
                type_ratio="none",
                num_processes=8,
            )
            self.assertIsNone(dataset.schedule)
            self.assertEqual(dataset[0]["_samtok_source_row_id"], 0)
            self.assertEqual(dataset[1]["_samtok_schedule_position"], 1)

    def test_stage1_eval_five_setting_contract(self):
        settings = parse_settings(["1", "s2", "3,4", "s5"])
        self.assertEqual(settings, list(SETTING_SPECS[:5]))
        self.assertEqual(
            [setting.cot_mode for setting in settings],
            ["disabled", "disabled", "disabled", "online", "ground_truth"],
        )
        self.assertEqual(
            [setting.stage1_te_lora for setting in settings],
            [False, False, True, True, True],
        )

        row = {"prompt": "Turn the cat blue", "mt_cot": to_cot([])}
        common = {
            "seed": 7,
            "num_inference_steps": 40,
            "cfg_scale": 4.0,
            "samtok_max_new_tokens": 128,
        }
        with mock.patch("run_eval.run_edit", return_value="output") as run:
            for setting in settings[1:]:
                with self.subTest(setting=setting.key):
                    run.reset_mock()
                    self.assertEqual(
                        run_samtok_setting(object(), setting, row, "source", common),
                        "output",
                    )
                    kwargs = run.call_args.kwargs
                    self.assertEqual(kwargs["seed"], 7)
                    self.assertEqual(kwargs["num_inference_steps"], 40)
                    self.assertEqual(kwargs["cfg_scale"], 4.0)
                    self.assertEqual(kwargs["samtok_max_new_tokens"], 128)
                    if setting.cot_mode == "online":
                        self.assertTrue(kwargs["enable_samtok_cot"])
                        self.assertIsNone(kwargs["mt_cot"])
                    elif setting.cot_mode == "ground_truth":
                        self.assertFalse(kwargs["enable_samtok_cot"])
                        self.assertEqual(kwargs["mt_cot"], row["mt_cot"])
                    else:
                        self.assertFalse(kwargs["enable_samtok_cot"])
                        self.assertIsNone(kwargs["mt_cot"])

    def test_stage1_eval_stock_call_uses_official_2511_arguments(self):
        pipe = mock.Mock(return_value="output")
        source = Image.new("RGB", (1235, 1024))
        self.assertEqual(stock_edit(pipe, source, "edit", 9, 40, 4.0), "output")
        args, kwargs = pipe.call_args
        self.assertEqual(args, ("edit",))
        self.assertEqual(kwargs["edit_image"], [source])
        self.assertEqual(kwargs["seed"], 9)
        self.assertEqual(kwargs["num_inference_steps"], 40)
        self.assertEqual(kwargs["cfg_scale"], 4.0)
        self.assertEqual((kwargs["height"], kwargs["width"]), (1024, 1235))
        self.assertTrue(kwargs["edit_image_auto_resize"])
        self.assertTrue(kwargs["zero_cond_t"])

    def test_stage2_eval_three_setting_contract(self):
        settings = parse_settings(["6", "s7", "s8_stage2_gt_cot"])
        self.assertEqual(settings, list(SETTING_SPECS[5:]))
        self.assertEqual(
            [setting.cot_mode for setting in settings],
            ["disabled", "online", "ground_truth"],
        )
        self.assertEqual(
            [setting.stage1_te_lora for setting in settings],
            [True, True, True],
        )
        self.assertEqual(
            [setting.number for setting in settings],
            [6, 7, 8],
        )

        row = {"prompt": "Turn the cat blue", "mt_cot": to_cot([])}
        common = {
            "seed": 7,
            "num_inference_steps": 40,
            "cfg_scale": 4.0,
            "samtok_max_new_tokens": 128,
        }
        with mock.patch("run_eval.run_edit", return_value="output") as run:
            for setting in settings:
                with self.subTest(setting=setting.key):
                    run.reset_mock()
                    self.assertEqual(
                        run_samtok_setting(object(), setting, row, "source", common),
                        "output",
                    )
                    kwargs = run.call_args.kwargs
                    if setting.number == 6:
                        self.assertFalse(kwargs["enable_samtok_cot"])
                        self.assertIsNone(kwargs["mt_cot"])
                    elif setting.number == 7:
                        self.assertTrue(kwargs["enable_samtok_cot"])
                        self.assertIsNone(kwargs["mt_cot"])
                    else:
                        self.assertFalse(kwargs["enable_samtok_cot"])
                        self.assertEqual(kwargs["mt_cot"], row["mt_cot"])

    def test_stage1_eval_metadata_validation_checks_all_rows_before_slicing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            images = folder / "images"
            images.mkdir()
            for name in ["source0.png", "target0.png", "source1.png", "target1.png"]:
                Image.new("RGB", (32, 32), "white").save(images / name)
            rows = [
                {
                    "image": f"images/target{index}.png",
                    "edit_image": f"images/source{index}.png",
                    "prompt": "Make the object blue",
                    "sample_type": "edit_mt",
                    "mt_cot": to_cot([]),
                    "provenance": {
                        "source_parquet": "color_00000.parquet",
                        "row_idx": index,
                        "edit_type": "color",
                    },
                }
                for index in range(2)
            ]
            rows[1]["prompt"] = "Make the object blue " + chr(0x84DD)
            metadata = folder / "validation.jsonl"
            metadata.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "English/ASCII-only"):
                load_and_validate_rows(metadata, folder, max_samples=1)

    def test_wandb_is_default_and_requires_explicit_account(self):
        with mock.patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test-key",
                "WANDB_ENTITY": "test-entity",
                "WANDB_PROJECT": "test-project",
            },
            clear=True,
        ):
            args = samtok_parser().parse_args(["--dataset_base_path", "/tmp"])
            self.assertTrue(args.enable_wandb_log)
            self.assertEqual(args.wandb_project, "test-project")
            validate_wandb_credentials(args)

        with mock.patch.dict(os.environ, {}, clear=True):
            args = samtok_parser().parse_args(["--dataset_base_path", "/tmp"])
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                validate_wandb_credentials(args)

            offline_args = samtok_parser().parse_args(
                ["--dataset_base_path", "/tmp", "--disable_wandb_log"]
            )
            self.assertFalse(offline_args.enable_wandb_log)
            validate_wandb_credentials(offline_args)

            process_args = samtok_parser().parse_args(
                ["--dataset_base_path", "/tmp", "--task", "sft:data_process"]
            )
            self.assertTrue(process_args.enable_wandb_log)
            validate_wandb_credentials(process_args)

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

    def test_eight_rank_schedule_matches_accelerate_stride(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for sample_type, count in [
                ("edit_mt", 17),
                ("edit_ntp", 9),
                ("edit", 9),
            ]:
                for index in range(count):
                    row = {
                        "prompt": f"{sample_type}-{index}",
                        "sample_type": sample_type,
                    }
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
                data_file_keys=[],
                type_ratio="edit_mt:2,edit_ntp:1,edit:1",
                num_processes=8,
                gradient_accumulation_steps=4,
                seed=11,
            )
            scheduled = [
                dataset.data[index]["sample_type"] for index in dataset.schedule
            ]
            for step_start in range(0, len(scheduled), 32):
                optimizer_step = scheduled[step_start : step_start + 32]
                self.assertEqual(
                    Counter(optimizer_step),
                    Counter(edit_mt=16, edit_ntp=8, edit=8),
                )
                rank_sequences = [optimizer_step[rank::8] for rank in range(8)]
                self.assertTrue(
                    all(sequence == rank_sequences[0] for sequence in rank_sequences)
                )
            first = dataset[0]
            self.assertEqual(first["_samtok_schedule_position"], 0)
            self.assertEqual(first["_samtok_source_row_id"], dataset.schedule[0])

    def test_shifted_cot_supervision_uses_template_last_position(self):
        hidden = torch.arange(16).reshape(1, 8, 2)
        labels = torch.tensor([[101, 102, 103]])
        shifted = shifted_cot_supervision(hidden, labels, template_length=4)
        self.assertTrue(torch.equal(shifted, hidden[:, 3:6]))
        self.assertEqual(shifted.shape[:2], labels.shape)
        with self.assertRaisesRegex(ValueError, "Invalid shifted CoT slice"):
            shifted_cot_supervision(hidden, labels, template_length=8)

    def test_stage1_loss_dispatch_and_weights(self):
        pipe = SimpleNamespace()
        ntp = torch.tensor(2.0, requires_grad=True)
        fm = torch.tensor(3.0, requires_grad=True)
        with mock.patch.object(loss_module, "SamtokNTPLoss", return_value=ntp), mock.patch.object(
            loss_module, "FlowMatchSFTLoss", return_value=fm
        ):
            mt_loss = loss_module.SamtokEditingLoss(
                pipe,
                sample_type="edit_mt",
                ntp_weight=0.5,
                fm_weight=2.0,
            )
            self.assertEqual(mt_loss.item(), 7.0)
            self.assertEqual(set(pipe.last_loss_log), {"loss_ntp", "loss_fm"})
            self.assertEqual(pipe.last_loss_debug["loss_total_dtype"], "float32")

            ntp_loss = loss_module.SamtokEditingLoss(
                pipe, sample_type="edit_ntp", ntp_weight=0.5, fm_weight=2.0
            )
            self.assertEqual(ntp_loss.item(), 1.0)
            self.assertEqual(set(pipe.last_loss_log), {"loss_ntp"})

            fm_loss = loss_module.SamtokEditingLoss(
                pipe, sample_type="edit", ntp_weight=0.5, fm_weight=2.0
            )
            self.assertEqual(fm_loss.item(), 6.0)
            self.assertEqual(set(pipe.last_loss_log), {"loss_fm"})

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

    def test_gres_edit_templates_are_english_only(self):
        for template in EDIT_VERB_TEMPLATES + GLOBAL_TEMPLATES:
            self.assertTrue(template.isascii(), template)

    def test_global_sampling_is_reproducible_and_not_prefix_based(self):
        candidates = [("shard.parquet", index, "add") for index in range(20)]
        first = sample_candidates(candidates, 5, seed=17)
        second = sample_candidates(candidates, 5, seed=17)
        self.assertEqual(first, second)
        self.assertNotEqual(first, candidates[:5])
        self.assertEqual(len(set(first)), 5)

        rows = [{"id": index} for index in range(20)]
        sampled = select_source_rows(rows, None, 5, seed=17)
        self.assertEqual(sampled, select_source_rows(rows, None, 5, seed=17))
        self.assertNotEqual(sampled, rows[:5])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            select_source_rows(rows, 5, 5, seed=17)

        pairs = [(Path(str(index)), Path(str(index))) for index in range(11)]
        partitions = [partition_pairs(pairs, 3, index) for index in range(3)]
        self.assertEqual(sum((partitions[index] for index in range(3)), []), pairs[::3] + pairs[1::3] + pairs[2::3])
        self.assertEqual(
            {pair for partition in partitions for pair in partition}, set(pairs)
        )

    def test_training_metadata_source_identity_exclusion(self):
        provenance_row = {
            "sample_type": "edit_mt",
            "provenance": {
                "source_parquet": "add_00001.parquet",
                "row_idx": 17,
            },
        }
        edit_row = {
            "sample_type": "edit",
            "edit_image": "images/remove_00002/000031_source.jpg",
        }
        self.assertEqual(
            source_identity_from_row(provenance_row),
            ("add_00001.parquet", 17),
        )
        self.assertEqual(
            source_identity_from_row(edit_row),
            ("remove_00002.parquet", 31),
        )
        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / "exclude.jsonl"
            metadata.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [provenance_row, edit_row, provenance_row]
                )
                + "\n",
                encoding="utf-8",
            )
            excluded, stats = load_excluded_source_ids([metadata])
        self.assertEqual(
            excluded,
            {("add_00001.parquet", 17), ("remove_00002.parquet", 31)},
        )
        self.assertEqual(stats["source_identities"], 2)
        self.assertEqual(stats["duplicate_identities"], 1)
        self.assertEqual(
            source_identity_from_row(
                {
                    "provenance": {
                        "source_parquet": "background change_00059.parquet",
                        "row_idx": 89,
                    }
                }
            ),
            ("background_change_00059.parquet", 89),
        )

    def test_qwen_image_block_forwards_kv_cache(self):
        class AttentionProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_cache = None

            def forward(self, image, text, kv_cache=None, **kwargs):
                self.kv_cache = kv_cache
                return torch.zeros_like(image), torch.zeros_like(text)

        block = QwenImageTransformerBlock(
            dim=4,
            num_attention_heads=1,
            attention_head_dim=4,
        )
        probe = AttentionProbe()
        block.attn = probe
        cache = (torch.zeros(1), torch.ones(1))
        block(
            image=torch.zeros(1, 2, 4),
            text=torch.zeros(1, 3, 4),
            temb=torch.zeros(1, 4),
            kv_cache=cache,
        )
        self.assertIs(probe.kv_cache, cache)

    def test_samtok_trainer_accepts_sharded_model_paths(self):
        trainer = QwenImageSamtokTrainingModule.__new__(
            QwenImageSamtokTrainingModule
        )
        configs = trainer.parse_model_configs(
            json.dumps([["text-00001.safetensors", "text-00002.safetensors"], "vae.safetensors"]),
            None,
        )
        self.assertEqual(
            configs[0].path,
            ["text-00001.safetensors", "text-00002.safetensors"],
        )
        self.assertEqual(configs[1].path, "vae.safetensors")


if __name__ == "__main__":
    unittest.main()
