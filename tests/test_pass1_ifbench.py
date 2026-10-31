"""Public IF-Bench adaptation; optional benchmark checkout supplies its verifier."""

import copy

import pytest

public_api = pytest.importorskip("samtok_benchmark.ifbench.public")

from samtok_edit21.pass1.evaluation.ifbench import adapt  # noqa: E402

SPAN = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"


def public_request():
    request = {
        "schema": public_api.SCHEMA,
        "protocol": public_api.PROTOCOL,
        "request_id": "fixture--S5",
        "case_id": "fixture",
        "setting": "S5",
        "source": {
            "path": "/not-read/source.jpg",
            "sha256": "0" * 64,
            "size": [100, 80],
        },
        "units": [
            {
                "id": "U1",
                "operation": "remove",
                "instruction": "Remove the selected region.",
                "has_ref": False,
                "locator": "point",
                "region": {"coordinates": [20, 40]},
            },
            {
                "id": "U2",
                "operation": "replace",
                "instruction": "Replace the selected region with a stool.",
                "has_ref": False,
                "locator": "box",
                "region": {"coordinates": [10, 20, 50, 80]},
            },
            {
                "id": "U3",
                "operation": "attribute",
                "instruction": "Make the selected region blue.",
                "has_ref": False,
                "locator": "mask",
                "region": {"path": "/not-read/public-mask.png", "sha256": "1" * 64},
            },
            {
                "id": "U4",
                "operation": "remove",
                "instruction": "Remove the red car on the left.",
                "has_ref": True,
                "locator": "none",
                "region": None,
            },
        ],
        "manifest_sha256": public_api.MANIFEST_SHA256,
    }
    request["input_digest"] = public_api.public_digest(request)
    return request


def test_mixed_public_request_only_looks_up_explicit_mask():
    class Cache(dict):
        reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return SPAN

    public = public_request()
    snapshot = copy.deepcopy(public)
    cache = Cache()
    request = adapt(public, cache)
    assert public == snapshot
    assert cache.reads == [("0" * 64, "1" * 64)]
    assert [t["label"] for t in request["targets"]] == ["U1", "U2", "U3", "U4"]
    assert request["targets"][0]["region"]["point"] == [200, 500]
    assert request["targets"][1]["region"]["box"] == [100, 250, 500, 1000]
    assert "region" not in request["targets"][3]


def test_private_field_is_rejected_even_with_recomputed_public_digest():
    public = public_request()
    public["units"][0]["target"] = {"mask": "/private/gt.png"}
    public["input_digest"] = public_api.public_digest(public)
    with pytest.raises(ValueError, match="Private/unknown/missing"):
        adapt(public, {})
