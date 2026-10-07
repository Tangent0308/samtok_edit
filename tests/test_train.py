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


def test_type_weight_schemes():
    from samtok_edit21.data.io import type_probabilities

    sizes = {"add": 150, "remove": 300, "replace": 200, "attribute": 300, "action": 30, "text": 20}  # 1,000 rows
    natural = type_probabilities("edit_ntp", sizes, "natural")
    assert natural["remove"] == pytest.approx(0.3) and natural["text"] == pytest.approx(0.02)
    main4 = type_probabilities("edit_umt:noref", sizes, "main4")
    assert main4["action"] == pytest.approx(0.03) and main4["text"] == pytest.approx(0.02)
    assert main4["add"] == pytest.approx(0.95 * 14 / 62) and main4["attribute"] == pytest.approx(0.95 * 20 / 62)
    v1 = type_probabilities("edit_ntp", sizes, "v1")
    assert v1["action"] == pytest.approx(10 / 82)
    # Plain and replay pools stay natural under every scheme.
    assert type_probabilities("edit", sizes, "main4") == type_probabilities("edit", sizes, "natural")
    rows = [_row("edit_ntp", t) for t, n in sizes.items() for _ in range(n)] + [_row("rec_ntp", "remove")] * 50
    _, report = make_schedule(rows, "stage1", 4, 8, steps=20, seed=0, type_weights="natural")
    assert report["type_weights"] == "natural"
    assert report["type_probabilities"]["edit_ntp"]["remove"] == pytest.approx(0.3, abs=1e-4)
    with pytest.raises(ValueError):
        make_schedule(rows, "stage1", 4, 8, steps=1, type_weights="uniform")


def test_binding_is_a_stage2_option():
    import argparse
    from samtok_edit21.training.engine import normalize_args

    args = argparse.Namespace(command="train", output="x", stage="stage1", binding="bias_span")
    with pytest.raises(ValueError, match="Stage 2"):
        normalize_args(args)
    args = normalize_args(argparse.Namespace(command="train", output="x", stage="stage2",
                                             binding="bias_clause", binding_beta=2.0, binding_eps=0.0))
    assert args.binding_config == {"mode": "bias_clause", "beta": 2.0, "eps": 0.0, "rank": 64}
    assert (args.accumulation, args.rank, args.lr, args.type_weights) == (4, 32, 1e-4, "main4")
    stage1 = normalize_args(argparse.Namespace(command="train", output="x", stage="stage1"))
    assert stage1.type_weights == "natural"


def test_stage2_takes_exactly_one_input_source():
    from samtok_edit21.training.engine import main

    for source in ([], ["--cache", "c", "--metadata", "m.jsonl"]):
        with pytest.raises(SystemExit, match="exactly one of --cache or --metadata"):
            main(["train", "--stage", "stage2", "--output", "x", "--steps", "1", "--save-steps", "4", *source])


def test_audit_accepts_only_the_runs_raw_te_conditioning(tmp_path):
    import importlib.util
    from pathlib import Path
    from samtok_edit21.data.provenance import FORMAT, PREPROCESSING

    spec = importlib.util.spec_from_file_location(
        "audit_run", Path(__file__).resolve().parents[1] / "scripts/diagnostics/audit_run.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    identity = {"schema": FORMAT, "preprocessing": PREPROCESSING, "te_adapter": None,
                "binding": {"schema": "b"}, "max_pixels": 1024, "metadata_sha256": "abc"}
    adapter = tmp_path / "stage2" / "adapter"
    adapter.mkdir(parents=True)
    run = {"max_pixels": 1024}
    for change, ok in (({}, True), ({"te_adapter": {"sha256": "s"}}, False),
                       ({"max_pixels": 2048}, False), ({"metadata_sha256": "other"}, False)):
        (adapter / "adapter.json").write_text(json.dumps({"conditioning_identity": {**identity, **change}}))
        if ok:
            assert audit.audit_on_the_fly(tmp_path, run, {"stage2.jsonl": "abc"})["mode"] == "on_the_fly"
        else:
            with pytest.raises(ValueError):
                audit.audit_on_the_fly(tmp_path, run, {"stage2.jsonl": "abc"})


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
