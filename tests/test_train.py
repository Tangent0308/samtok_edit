import json

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import Dataset, SequentialSampler

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_data_process_task, launch_training_task
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from samtok_edit21.data import make_schedule, row_kind
from samtok_edit21.train import ScheduledMetadata, verify_cache
from samtok_edit21.data import file_hash, row_hash


def _row(kind):
    sample_type, _, variant = kind.partition(":")
    row = {
        "sample_type": sample_type,
        "edit_type": "attribute",
        "edit_image": "source.png",
        "image": "target.png",
        "prompt": "Change the object.",
    }
    if sample_type == "edit_ntp":
        row.pop("image")
        row["mt_cot"] = "```json\n[{\"mask_2d\": \"<|mt_start|><|mt_0000|><|mt_0256|><|mt_end|>\", \"label\": \"object\"}]\n```"
    if sample_type == "edit_umt":
        row["instr_variant"] = variant
    return row


def test_schedule_is_exact_on_each_rank_when_local_accumulation_is_a_block():
    kinds = [
        "edit_ntp",
        "edit_ntp",
        "edit_ntp",
        "edit_umt:ref",
        "edit_umt:ref",
        "edit_umt:noref",
        "edit_umt:noref",
        "edit",
    ]
    rows = [_row(kind) for kind in kinds]
    schedule, report = make_schedule(rows, "stage1", 2, 8, steps=1, seed=0)
    assert report["per_step"] == {
        "edit_ntp": 6,
        "edit_umt:ref": 4,
        "edit_umt:noref": 4,
        "edit": 2,
    }
    for rank in range(2):
        local = [row_kind(rows[schedule[i]]) for i in range(rank, len(schedule), 2)]
        assert {kind: local.count(kind) for kind in set(local)} == {
            "edit_ntp": 3,
            "edit_umt:ref": 2,
            "edit_umt:noref": 2,
            "edit": 1,
        }


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


def test_cache_manifest_checks_geometry_and_checksum(tmp_path):
    row = {
        "sample_type": "edit",
        "edit_type": "attribute",
        "edit_image": "source.png",
        "image": "target.png",
        "prompt": "Change the object.",
    }
    shard = tmp_path / "0" / "0.pth"
    shard.parent.mkdir()
    inputs = {
        "input_latents": torch.zeros(1, 64, 4, 4),
        "edit_latents": [torch.zeros(1, 64, 4, 4)],
        "prompt_embeds": torch.zeros(1, 4, 4096),
        "prompt_embeds_mask": torch.ones(1, 4, dtype=torch.bool),
        "edit_image_pad_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    torch.save(inputs, shard)
    manifest = {
        "format": "samtok21-cache-v1",
        "rows": [{**row, "_cache_file": "0/0.pth"}],
    }
    side = {
        "row_hash": row_hash(row),
        "sha256": file_hash(shard),
    }
    (shard.with_suffix(".json")).write_text(json.dumps(side))
    assert verify_cache(tmp_path, manifest)
    shard.write_bytes(b"corrupt")
    try:
        verify_cache(tmp_path, manifest)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupted cache was accepted")
