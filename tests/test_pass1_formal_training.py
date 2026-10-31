import unittest

import torch

from samtok_edit21.pass1.formal_materialize import conflicting_requests
from samtok_edit21.pass1.formal_prepare import pack_batches
from samtok_edit21.pass1.formal_train import lr_for, rank_samples


class FormalTrainingTests(unittest.TestCase):
    def test_partial_batch_matches_mean_gradient_of_real_requests(self):
        batches = [{"indices": [0, 1, 2, 3, 4, -1]}]
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = (2 * x * x).mean().item()
        observed = 0.0
        for rank in range(2):
            p = torch.tensor(1.0, requires_grad=True)
            for index, weight in rank_samples(batches, 0, 1, rank, 2, 3):
                ((p * x[index]) ** 2 * weight / 3).backward()
            observed += p.grad.item() / 2
        self.assertAlmostEqual(expected, observed, places=5)

    def test_packing_preserves_pairs_and_uses_every_request_once(self):
        rows = [
            {"pair_id": "a"},
            {"pair_id": "a"},
            {},
            {},
            {"pair_id": "b"},
            {"pair_id": "b"},
            {},
            {},
            {},
        ]
        packed = pack_batches(rows, list(range(len(rows))), 4, 7, "main")
        locations = {i: j for j, b in enumerate(packed) for i in b["indices"] if i >= 0}
        self.assertEqual(sorted(locations), list(range(len(rows))))
        self.assertEqual(locations[0], locations[1])
        self.assertEqual(locations[4], locations[5])
        self.assertTrue(all(len(b["indices"]) == 4 for b in packed))

    def test_missing_partner_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            pack_batches([{"pair_id": "orphan"}], [0], 4, 7, "main")

    def test_resume_slice_does_not_repeat_previous_updates(self):
        batches = [{"indices": [0, 1, 2, 3]}, {"indices": [4, 5, 6, 7]}]
        self.assertEqual(rank_samples(batches, 1, 2, 1, 2, 2), [(5, 1.0), (7, 1.0)])

    def test_learning_rate_uses_global_update(self):
        self.assertAlmostEqual(lr_for(2500, 2500, 2e-5), 2e-6)
        self.assertLess(lr_for(1, 2500, 2e-5), lr_for(75, 2500, 2e-5))
        self.assertGreater(lr_for(500, 2500, 2e-5), lr_for(1500, 2500, 2e-5))

    def test_conflicts_quarantine_every_answer_variant_ignoring_label(self):
        def row(id, label, span):
            return {
                "id": id,
                "image_sha256": "image",
                "request": {
                    "targets": [
                        {
                            "label": label,
                            "instruction": "Remove the selected region.",
                            "has_ref": False,
                            "locator": "point",
                            "region": {"point": [100, 200]},
                        }
                    ]
                },
                "answer": [{"label": label, "mask_2d": span}],
            }

        bad, events = conflicting_requests(
            [row("a", "U1", "x"), row("b", "U3", "y"), row("c", "U2", "x")]
        )
        self.assertEqual(bad, {"a", "b", "c"})
        self.assertTrue(events)


if __name__ == "__main__":
    unittest.main()
