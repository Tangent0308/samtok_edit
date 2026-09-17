import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in [
    REPO_ROOT / "DiffSynth-Studio",
    REPO_ROOT / "scripts" / "eval",
]:
    sys.path.insert(0, str(path))

from diffsynth.models.qwen_image_dit import QwenDoubleStreamAttention  # noqa: E402
from consolidate_mask_token_interpretability import summarize_metrics  # noqa: E402
from prepare_mask_token_interventions import (  # noqa: E402
    ADDITIONAL_SELECTIONS,
    CORE_SELECTIONS,
    SELECTION_SETS,
)
from run_mask_token_interpretability import MaskTokenAttentionProbe  # noqa: E402
from summarize_mask_token_interpretability import (  # noqa: E402
    attention_metrics,
    top_area_mask,
)
from visualize_mask_token_attention_clear import (  # noqa: E402
    attention_scale,
    render_attention_focus,
    render_mask_overlay,
)


class MaskTokenInterventionTest(unittest.TestCase):
    def test_selection_is_small_unique_and_same_case_not_reused(self):
        self.assertEqual(len(CORE_SELECTIONS), 3)
        self.assertEqual(len(ADDITIONAL_SELECTIONS), 9)
        combined = SELECTION_SETS["all"]
        self.assertEqual(len(combined), 12)
        self.assertEqual(len({item.benchmark_index for item in combined}), 12)
        self.assertEqual(len({item.semantic_category for item in combined}), 12)

    def test_additional_selection_balances_operations_and_categories(self):
        self.assertEqual(
            [item.semantic_category for item in ADDITIONAL_SELECTIONS],
            [
                "fish",
                "turtle",
                "monkey",
                "duck",
                "chicken",
                "horse",
                "dog",
                "giraffe",
                "rabbit",
            ],
        )

    def test_top_area_mask_has_exact_requested_area(self):
        heatmap = np.arange(16, dtype=np.float32).reshape(4, 4)
        selected = top_area_mask(heatmap, 5)
        self.assertEqual(int(selected.sum()), 5)
        self.assertTrue(selected[3, 3])
        self.assertFalse(selected[0, 0])

    def test_attention_metrics_rewards_the_selected_instance(self):
        target = np.zeros((4, 4), dtype=bool)
        target[:, :2] = True
        other = ~target
        heatmaps = np.zeros((2, 4, 4), dtype=np.float32)
        heatmaps[:, :, :2] = 1
        heatmaps /= heatmaps.sum(axis=(1, 2), keepdims=True)
        metrics = attention_metrics(heatmaps, target, target, other, other)
        self.assertGreater(metrics["decoded_routing_margin"], 0.99)
        self.assertGreater(metrics["decoded_density_routing_margin"], 1.99)
        self.assertEqual(metrics["decoded_target_top_area_iou_mean"], 1.0)
        self.assertEqual(metrics["decoded_target_top_area_iou_chance"], 1 / 3)
        self.assertEqual(metrics["decoded_target_top_area_iou_lift"], 3.0)
        self.assertEqual(metrics["decoded_target_peak_inside_rate"], 1.0)

    def test_probe_extracts_normalized_source_attention(self):
        torch.manual_seed(1)
        attention = QwenDoubleStreamAttention(
            dim_a=8,
            dim_b=8,
            num_heads=2,
            head_dim=4,
        )
        probe = MaskTokenAttentionProbe(layer_ids=[0], step_ids=[0])
        probe.mask_positions = [1, 2]
        probe.current = {
            "progress_id": 0,
            "timestep": 1.0,
            "source_offset": 4,
            "source_shape": (2, 2),
        }
        probe._pre_hook(
            0,
            attention,
            (),
            {
                "image": torch.randn(1, 8, 8),
                "text": torch.randn(1, 3, 8),
                "image_rotary_emb": None,
                "attention_mask": None,
            },
        )
        self.assertEqual(len(probe.records), 1)
        record = probe.records[0]
        self.assertEqual(record["mask_query_to_source_heatmap"].shape, (2, 2))
        self.assertEqual(record["source_query_to_mask_heatmap"].shape, (2, 2))
        self.assertAlmostEqual(
            float(record["mask_query_to_source_heatmap"].sum()), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(record["source_query_to_mask_heatmap"].sum()), 1.0, places=6
        )
        self.assertGreater(record["raw_source_attention_mass"], 0.0)
        self.assertLess(record["raw_source_attention_mass"], 1.0)
        self.assertGreater(record["raw_mask_key_attention_mass"], 0.0)
        self.assertLess(record["raw_mask_key_attention_mass"], 1.0)

    def test_clear_attention_renderer_preserves_requested_geometry(self):
        source = Image.new("RGB", (32, 32), "gray")
        heatmap = np.ones((4, 4), dtype=np.float32) / 16
        scale = attention_scale(heatmap[None], heatmap[None])
        self.assertEqual(scale, 2.0)
        focused = render_attention_focus(source, heatmap, scale, (96, 96))
        self.assertEqual(focused.size, (96, 96))

    def test_mask_overlay_supports_raw_and_decoded_masks_with_same_renderer(self):
        source = Image.new("RGB", (16, 16), (100, 100, 100))
        mask = np.zeros((16, 16), dtype=bool)
        mask[:, :8] = True
        rendered = render_mask_overlay(
            source,
            mask,
            np.asarray([255, 55, 55], dtype=np.float32),
            (64, 64),
        )
        values = np.asarray(rendered)
        self.assertEqual(rendered.size, (64, 64))
        self.assertGreater(float(values[:, :24, 0].mean()), float(values[:, 40:, 0].mean()))

    def test_unified_summary_recomputes_both_attention_directions(self):
        def condition(margin, density, iou, chance, lift):
            return {
                direction: {
                    "decoded_routing_margin": margin,
                    "decoded_density_routing_margin": density,
                    "decoded_target_top_area_iou_mean": iou,
                    "decoded_target_top_area_iou_chance": chance,
                    "decoded_target_top_area_iou_lift": lift,
                }
                for direction in ("mask_query_to_source", "source_query_to_mask")
            }

        row = {
            "original": condition(0.2, 0.4, 0.3, 0.1, 3.0),
            "alternate": condition(-0.1, -0.2, 0.1, 0.1, 1.0),
            "paired": {
                "directions": {
                    direction: {
                        "attention_mask_shift_cosine": 0.5,
                        "counterfactual_switch_score": 0.25,
                    }
                    for direction in ("mask_query_to_source", "source_query_to_mask")
                }
            },
        }
        summary = summarize_metrics([row])
        for direction in ("mask_query_to_source", "source_query_to_mask"):
            values = summary[direction]
            self.assertEqual(values["total_conditions"], 2)
            self.assertEqual(
                values["conditions_with_positive_decoded_routing_margin"], 1
            )
            self.assertEqual(
                values["cases_with_both_decoded_routing_margins_positive"], 0
            )
            self.assertAlmostEqual(values["mean_decoded_target_top_area_iou"], 0.2)
            self.assertEqual(values["cases_with_positive_counterfactual_switch_score"], 1)


if __name__ == "__main__":
    unittest.main()
