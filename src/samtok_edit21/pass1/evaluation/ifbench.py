"""Public IF-Bench request -> frozen, trained samtok_pass1_v2 protocol.

This module never accepts a private manifest case or reads a GT mask.
Only an explicitly supplied mask locator can query the public codec cache.
"""

from samtok_benchmark.ifbench.public import verify

from samtok_edit21.pass1.protocol import (
    VERSION,
    box_normalized,
    point_normalized,
    valid_span,
    validate_request,
)


def adapt(public, public_mask_tokens):
    verify(public)
    targets = []
    for unit in public["units"]:
        locator = unit["locator"]
        target = {
            "label": unit["id"],
            "instruction": unit["instruction"],
            "has_ref": unit["has_ref"],
            "locator": locator,
        }
        if locator == "point":
            target["region"] = {
                "point": point_normalized(
                    unit["region"]["coordinates"], public["source"]["size"]
                )
            }
        elif locator == "box":
            target["region"] = {
                "box": box_normalized(
                    unit["region"]["coordinates"], public["source"]["size"]
                )
            }
        elif locator == "mask":
            key = (public["source"]["sha256"], unit["region"]["sha256"])
            span = public_mask_tokens[key]
            if not valid_span(span):
                raise ValueError("Invalid public mask token")
            target["region"] = {"mask_2d": span}
        targets.append(target)
    return validate_request(
        {
            "protocol": VERSION,
            "coordinate_system": "normalized_0_1000",
            "targets": targets,
        }
    )
