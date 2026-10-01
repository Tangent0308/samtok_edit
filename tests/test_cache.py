import pytest

from samtok_edit21.data.io import file_hash, row_hash, write_json
from samtok_edit21.training.objectives import verify_cache


def test_cache_identity_and_content_corruption(tmp_path):
    row = dict(
        sample_type="edit",
        edit_type="attribute",
        edit_image="source.png",
        image="target.png",
        prompt="Turn the car blue.",
    )
    identity = {"te_adapter": None, "metadata_sha256": "example"}
    shard = tmp_path / "0" / "00000000.pt"
    shard.parent.mkdir()
    shard.write_bytes(b"cached tensor contents")
    manifest = {
        "format": "samtok21-cache-v1",
        "identity": identity,
        "rows": [{**row, "_cache_file": "0/00000000.pt"}],
    }
    side = {"row_hash": row_hash(row), "identity": identity, "sha256": file_hash(shard)}
    write_json(shard.with_suffix(".json"), side)
    verify_cache(tmp_path, manifest)
    manifest["rows"][0]["prompt"] = "Turn the car red."
    with pytest.raises(ValueError, match="stale"):
        verify_cache(tmp_path, manifest)
    manifest["rows"][0]["prompt"] = row["prompt"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_cache(tmp_path, manifest)


def test_geometry_requires_qwen3_vae_alignment():
    import torch
    from samtok_edit21.training.objectives import validate_conditioning

    inputs = {
        "input_latents": torch.zeros(1, 64, 4, 4),
        "prompt_embeds": torch.zeros(1, 8, 4096),
        "edit_image_pad_mask": torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
        ),
        "edit_latents": [torch.zeros(1, 64, 4, 4)],
    }
    validate_conditioning(inputs)
    inputs["edit_image_pad_mask"][0, 0] = False
    with pytest.raises(ValueError, match="disagree"):
        validate_conditioning(inputs)
