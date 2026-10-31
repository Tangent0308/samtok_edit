"""Checks for data-leakage boundaries and the actual new sampler failure modes."""

import copy
import unittest

import numpy as np

from samtok_edit21.pass1.formal_materialize import sample_box, sample_point
from samtok_edit21.pass1.formal_select import (
    add_counterfactual_pairs,
    canonicalize_teachers,
    distinct,
    eligible_modes,
    rebalance_modes,
)


class FormalDataTests(unittest.TestCase):
    def config(self):
        return {
            "seed": 123,
            "dense_parent_modes": ["noref_box", "noref_mask"],
            "known_host_fallback_modes": ["noref_box", "ref_mask", "noref_mask"],
            "main_modes": {
                "ref_none": 0.2,
                "ref_point": 0.2,
                "noref_point": 0.2,
                "ref_box": 0.1,
                "noref_box": 0.1,
                "ref_mask": 0.1,
                "noref_mask": 0.1,
            },
            "point_train": {"deep": 0.5, "uniform": 0.4, "inner_boundary": 0.1},
            "box_train": {"tight": 1.0},
            "counterfactual_main_row_fraction": 1.0,
        }

    def test_host_fallback_has_no_ambiguous_text_or_point_supervision(self):
        r = {
            "key": "M:r:U1",
            "operation": "add",
            "has_site": True,
            "ref_instruction": "Add a badge on the chest of the person.",
            "noref_eligible": True,
        }
        self.assertEqual(
            eligible_modes(r, self.config()), ["noref_box", "ref_mask", "noref_mask"]
        )

    def test_dense_category_never_enables_ref(self):
        r = {"key": "P:s:0", "category": "chair", "parts": []}
        self.assertEqual(
            eligible_modes(r, self.config()), ["noref_point", "noref_box", "noref_mask"]
        )
        r["parts"] = [1]
        self.assertNotIn("noref_point", eligible_modes(r, self.config()))

    def test_dedup_uses_both_region_and_code_but_allows_overlap(self):
        rows = [
            {"span": "a", "s_digest": "x"},
            {"span": "a", "s_digest": "y"},
            {"span": "b", "s_digest": "x"},
            {"span": "c", "s_digest": "z"},
        ]
        self.assertEqual(distinct(rows), [rows[0], rows[3]])

    def test_same_S_teacher_is_unified_without_changing_instruction_or_raw_span(self):
        a = {
            "source": "s",
            "s_digest": "same",
            "key": "M:a:U1",
            "span": "a",
            "roundtrip": 0.75,
            "ref_instruction": "Keep this original instruction.",
        }
        b = {
            "source": "s",
            "s_digest": "same",
            "key": "P:s:0",
            "span": "b",
            "roundtrip": 0.85,
        }
        conflicts = canonicalize_teachers([a, b])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(a["canonical_label_key"], b["key"])
        self.assertEqual(a["span"], "a")
        self.assertEqual(a["ref_instruction"], "Keep this original instruction.")

    def test_eval_click_does_not_use_teacher(self):
        mask = np.zeros((21, 21), bool)
        mask[4:17, 4:17] = True
        empty_teacher = np.zeros_like(mask)
        p, info = sample_point(
            mask, empty_teacher, [21, 21], policy="deep", seed=1, training=False
        )
        self.assertFalse(info["teacher_contains_point"])
        self.assertFalse(info["training_teacher_filter"])
        with self.assertRaisesRegex(ValueError, "no_teacher_consistent"):
            sample_point(
                mask, empty_teacher, [21, 21], policy="deep", seed=1, training=True
            )

    def test_quantization_can_reject_tiny_training_point(self):
        mask = np.zeros((2, 3000), bool)
        mask[0, 1] = True
        with self.assertRaisesRegex(ValueError, "quantization_safe"):
            sample_point(mask, mask, [3000, 2], policy="uniform", seed=1, training=True)

    def test_evaluation_box_cannot_use_jitter(self):
        with self.assertRaisesRegex(ValueError, "evaluation_box_jitter"):
            sample_box(
                [1, 2, 10, 20],
                [100, 100],
                policy="outward_jitter_5pct",
                seed=1,
                training=False,
            )

    def test_mode_allocation_compensates_dense_constraints(self):
        config = self.config()
        rows = []
        bank = {}
        for i in range(100):
            key = str(i)
            bank[key] = {
                "modes": ["noref_point", "noref_box", "noref_mask"]
                if i < 35
                else list(config["main_modes"])
            }
            rows.append({"targets": [{"region_key": key}]})
        rebalance_modes(rows, bank, config, "main")
        counts = {
            m: sum(r["targets"][0]["mode"] == m for r in rows)
            for m in config["main_modes"]
        }
        for m, weight in config["main_modes"].items():
            self.assertLessEqual(abs(counts[m] - 100 * weight), 1)

    def test_counterfactual_only_changes_one_region_and_keeps_context(self):
        config = self.config()
        a = {
            "key": "P:s:0",
            "source": "s",
            "index": 0,
            "span": "a",
            "s_digest": "a",
            "bbox": [0, 0, 5, 5],
            "same": [1],
            "modes": ["noref_point", "noref_box", "noref_mask"],
        }
        b = {
            **a,
            "key": "P:s:1",
            "index": 1,
            "span": "b",
            "s_digest": "b",
            "bbox": [6, 6, 10, 10],
            "same": [0],
        }
        c = {
            **a,
            "key": "P:s:2",
            "index": 2,
            "span": "c",
            "s_digest": "c",
            "bbox": [11, 11, 15, 15],
            "same": [],
        }
        plans = [
            {
                "id": str(i),
                "source": "s",
                "kind": "dense",
                "K": 2,
                "targets": [
                    {
                        "label": "U4",
                        "region_key": a["key"],
                        "mode": "noref_point",
                        "locator_policy": "deep",
                        "geometry_seed": 1,
                    },
                    {
                        "label": "U1",
                        "region_key": c["key"],
                        "mode": "noref_mask",
                        "locator_policy": "copy",
                        "geometry_seed": 2,
                    },
                ],
            }
            for i in range(2)
        ]
        info = add_counterfactual_pairs(
            plans, {r["key"]: r for r in (a, b, c)}, {"s": [a, b, c]}, config
        )
        self.assertEqual(info["selected_pairs"], 1)
        self.assertEqual(plans[0]["targets"][1], plans[1]["targets"][1])
        x, y = (
            copy.deepcopy(plans[0]["targets"][0]),
            copy.deepcopy(plans[1]["targets"][0]),
        )
        self.assertNotEqual(x.pop("region_key"), y.pop("region_key"))
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
