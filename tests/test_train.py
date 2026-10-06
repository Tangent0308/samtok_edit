import json

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import Dataset, SequentialSampler

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_data_process_task, launch_training_task
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from samtok_edit21.data.io import make_schedule, row_kind


def _row(kind, edit_type="attribute"):
    sample_type, _, variant = kind.partition(":")
    row = {"sample_type": sample_type, "edit_type": edit_type, "edit_image": "source.png",
           "image": "target.png", "prompt": "Change the object."}
    if sample_type in {"edit_ntp", "rec_ntp"}:
        row.pop("image")
        row["mt_cot"] = "cot"
    if sample_type == "edit_umt":
        row["instr_variant"] = variant
    return row


def test_stage1_schedule_is_ntp_with_exact_replay_ratio_on_each_rank():
    rows = [_row("edit_ntp", t) for t in ("add", "remove", "attribute")] + [_row("rec_ntp", "remove")]
    schedule, report = make_schedule(rows, "stage1", 4, 8, steps=3, seed=0)
    assert report["per_step"] == {"edit_ntp": 28, "rec_ntp": 4}
    for rank in range(4):
        local = [row_kind(rows[schedule[i]]) for i in range(rank, len(schedule), 4)]
        assert {kind: local.count(kind) for kind in set(local)} == {"edit_ntp": 21, "rec_ntp": 3}
    with pytest.raises(ValueError, match="cannot contain edit rows"):
        make_schedule(rows + [_row("edit")], "stage1", 1, 8, steps=1)


def test_stage2_schedule_keeps_ref_noref_plain_ratio():
    rows = [_row("edit_umt:ref"), _row("edit_umt:noref"), _row("edit")]
    schedule, report = make_schedule(rows, "stage2", 8, 4, steps=2, seed=0)
    assert report["per_step"] == {"edit_umt:ref": 8, "edit_umt:noref": 16, "edit": 8}
    with pytest.raises(ValueError, match="cannot contain edit_ntp rows"):
        make_schedule(rows + [_row("edit_ntp")], "stage2", 1, 4, steps=1)


def test_binding_is_a_stage2_option():
    import argparse
    from samtok_edit21.training.engine import normalize_args

    args = argparse.Namespace(command="train", output="x", stage="stage1", binding="bias_span")
    with pytest.raises(ValueError, match="Stage 2"):
        normalize_args(args)
    args = normalize_args(argparse.Namespace(command="train", output="x", stage="stage2",
                                             binding="bias_clause", binding_beta=2.0, binding_eps=0.0))
    assert args.binding_config == {"mode": "bias_clause", "beta": 2.0, "eps": 0.0, "rank": 64}
    assert (args.accumulation, args.rank, args.lr) == (4, 32, 1e-4)


class _TinyDataset(Dataset):
    load_from_cache = False

    def __init__(self):
        self.x = list(range(4))
        self.schedule_sampler = SequentialSampler(self)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": torch.tensor([float(self.x[index] + 1)])}


class _TinyModel(DiffusionTrainingModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, data):
        return ((self.weight - data["x"]) ** 2).mean()

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return state_dict


def test_runner_accepts_project_sampler_and_checkpoint(tmp_path):
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    output = tmp_path / "runner"
    launch_training_task(
        accelerator,
        _TinyDataset(),
        _TinyModel(),
        ModelLogger(str(output), enable_csv_log=True),
        learning_rate=0.1,
        weight_decay=0.0,
        num_workers=0,
        save_steps=10,
        num_epochs=1,
        args=None,
    )
    assert (output / "loss.csv").exists()
    assert (output / "step-4.safetensors").exists()


def test_data_process_runner_accepts_explicit_defaults(tmp_path):
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(even_batches=False))
    output = tmp_path / "cache"
    launch_data_process_task(
        accelerator,
        _TinyDataset(),
        _TinyModel(),
        ModelLogger(str(output)),
        num_workers=0,
        args=None,
    )
    assert (output / "0" / "0.pth").exists()
