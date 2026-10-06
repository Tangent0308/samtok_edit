from collections import Counter

import pytest

from samtok_edit21.data.protocol import (
    REC_TEMPLATE,
    Unit,
    box_coords,
    box_of,
    boxes_in,
    pixel_box,
    regions_in,
    relative_box,
    span_of,
    spans_in,
    to_cot,
    parse_cot,
    parse_generated_cot,
    render_units,
    grouped_units,
    interactive_prompt,
    validate_inline,
    validate_row,
)
from samtok_edit21.preparation.converters import convert_record
from samtok_edit21.data.io import make_schedule, row_kind, capped_plain_weights

A, B = span_of([3, 300]), span_of([40, 450])
X, Y = box_of((100, 200, 300, 400)), box_of((0, 0, 1000, 1000))


def test_two_codebooks_and_strict_generation():
    for codes in ([256, 300], [10, 255], [1, 512]):
        with pytest.raises(ValueError):
            span_of(codes)
    with pytest.raises(ValueError):
        spans_in("<|mt_start|><|mt_0001|><|mt_end|>")
    with pytest.raises(ValueError):
        parse_cot('[{"mask_2d":"' + A + '","label":"cat","other":1}]')
    with pytest.raises(ValueError):
        parse_cot("[]", nonempty=True)
    assert parse_generated_cot("<think>\n\n</think>\n" + to_cot([(A, "cat")])) == [
        (A, "cat")
    ]
    with pytest.raises(ValueError):
        parse_generated_cot("<think>unterminated" + to_cot([(A, "cat")]))


def test_quoted_text_labels_preserved_and_tokens_adjacent():
    label = 'text "OPEN NOW"'
    cot = to_cot([(A, label)])
    assert parse_cot(cot + "<|im_end|>") == [(A, label)]
    prompt = render_units(
        'Change the text "OPEN NOW" to "CLOSED".', [Unit(label, (A, B), "text")]
    )
    assert 'text "OPEN NOW" ' + A + B in prompt
    noref = render_units(
        'Change the text "OPEN NOW" to "CLOSED".',
        [Unit(label, (A,), "text")],
        variant="noref",
    )
    assert noref == "Change the text in this region " + A + ' to "CLOSED".'


def test_composite_simultaneous_targets_and_ambiguity():
    prompt = render_units(
        "Remove the cat and turn the dog blue.",
        [Unit("cat", (A,), "remove"), Unit("dog", (B,), "attribute")],
        variant="noref",
    )
    assert (
        prompt
        == "Remove the object in this region "
        + A
        + " and turn this region "
        + B
        + " blue."
    )
    with pytest.raises(ValueError):
        render_units("Move the cat near the cat.", [Unit("cat", (A,))])
    with pytest.raises(ValueError):
        render_units(
            "Color the large cat.", [Unit("large cat", (A,)), Unit("cat", (B,))]
        )


def test_add_preserves_new_object_and_global_is_nonempty():
    prompt = render_units(
        "Add a red ball next to the chair.",
        [Unit("a red ball next to the chair", (A,), "add", "next to the chair")],
        variant="noref",
    )
    assert prompt == "Add a red ball in this region " + A + "."
    boxed = render_units(
        "Add a red ball next to the chair.",
        [Unit("a red ball next to the chair", (X,), "add", "next to the chair")],
        variant="noref",
    )
    assert boxed == "Add a red ball in this region " + X + "."
    assert interactive_prompt("Add a red ball.", [[X]]) == "Add a red ball in this region " + X + "."
    with pytest.raises(ValueError):
        render_units(
            "Add a ball on the table near the chair.",
            [Unit("a ball on the table near the chair", (A,), "add", "on the table")],
            variant="noref",
        )
    assert (
        render_units("Apply a watercolor style.", [Unit("this image", (A,), "global")])
        == "Apply a watercolor style to this image " + A + "."
    )


def test_interactive_regions_follow_text_order():
    assert (
        interactive_prompt("Make this region blue and this region red.", [[A], [B]])
        == "Make this region " + A + " blue and this region " + B + " red."
    )
    with pytest.raises(ValueError):
        interactive_prompt("Make this region blue.", [[A], [B]])


def test_plural_labels_rebind_with_or_without_article():
    items = [(A, "one of the cats"), (B, "one of the cats")]
    assert (
        render_units("Remove cats.", grouped_units("Remove cats.", items))
        == "Remove cats " + A + B + "."
    )
    assert (
        render_units("Remove the cats.", grouped_units("Remove the cats.", items))
        == "Remove the cats " + A + B + "."
    )


def test_protocol_splits_and_forbids_legacy_supervision():
    record = dict(
        instruction="Turn the cat blue.",
        edit_image="source.png",
        image="target.png",
        units=[dict(ref_phrase="cat", mask_codes=[A], edit_type="attribute")],
    )
    rows, errors = convert_record(record)
    assert not errors and [r["sample_type"] for r in rows] == [
        "edit_ntp",
        "edit",
        "edit_umt",
        "edit_umt",
    ]
    assert "image" not in rows[0] and all("mt_cot" not in r for r in rows[1:])
    with pytest.raises(ValueError):
        validate_row({**rows[0], "image": "target.png"})
    with pytest.raises(ValueError):
        validate_row({**rows[1], "mt_cot": rows[0]["mt_cot"]})
    with pytest.raises(ValueError):
        validate_row({**rows[-1], "edit_image": ["a.png", "b.png"]})


def test_action_background_and_composite_unit_order():
    record = dict(
        instruction="Make the dog run and turn the background blue.",
        edit_image="source.png",
        image="target.png",
        units=[
            dict(ref_phrase="background", mask_codes=[B], edit_type="background"),
            dict(ref_phrase="dog", mask_codes=[A], edit_type="action"),
        ],
    )
    # Multi-unit rendering still binds in instruction order, but v2 metadata
    # excludes composite rows entirely.
    units = [Unit("dog", (A,), "action"), Unit("background", (B,), "background")]
    assert (
        render_units(record["instruction"], units, variant="noref")
        == "Make the object in this region "
        + A
        + " run and turn this region "
        + B
        + " blue."
    )
    with pytest.raises(ValueError, match="composite"):
        convert_record(record)


def test_box_tokens_are_canonical_integers():
    assert box_coords(X) == (100, 200, 300, 400) and boxes_in("Add a cat " + X + Y) == [X, Y]
    for bad in ("<|box_start|>[100, 200, 100, 400]<|box_end|>",
                "<|box_start|>[0100, 200, 300, 400]<|box_end|>",
                "<|box_start|>[100,200,300,400]<|box_end|>",
                "<|box_start|>[100, 200, 300, 1001]<|box_end|>"):
        with pytest.raises(ValueError):
            boxes_in("Add a cat " + bad)
    with pytest.raises(ValueError):
        box_of((1.0, 2, 3, 4))
    assert regions_in("Add a cat " + X + " and remove " + A) == [X, A]
    # Outward rounding of exclusive pixel bounds keeps a nonempty box.
    assert relative_box((0, 0, 1, 1), 3000, 3000) == (0, 0, 1, 1)
    assert relative_box((1, 5, 5, 8), 12, 10) == (83, 500, 417, 800)
    import numpy as np
    mask = np.zeros((10, 12), bool)
    mask[5:8, 1:5] = True
    assert pixel_box(mask) == (83, 500, 417, 800)


def test_bbox_localization_round_trip_and_type_kinds():
    cot = to_cot([(X, "red ball next to the chair")])
    assert '"bbox_2d": [100, 200, 300, 400]' in cot
    assert parse_cot(cot + "<|im_end|>") == [(X, "red ball next to the chair")]
    for bad in ('[{"bbox_2d": [100, 200, 300], "label": "x"}]',
                '[{"bbox_2d": [100.0, 200, 300, 400], "label": "x"}]',
                '[{"bbox_2d": "' + A.replace('"', '') + '", "label": "x"}]',
                '[{"bbox_2d": [100, 200, 300, 400], "mask_2d": "' + A + '", "label": "x"}]'):
        with pytest.raises(ValueError):
            parse_cot(bad)
    with pytest.raises(ValueError):
        grouped_units("Remove the cats.", [(A, "one of the cats"), (X, "one of the cats")])
    add = dict(sample_type="edit_ntp", edit_type="add", edit_image="s.png",
               prompt="Add a red ball next to the chair.", mt_cot=cot)
    assert validate_row(add) is add
    with pytest.raises(ValueError, match="box"):
        validate_row({**add, "mt_cot": to_cot([(A, "red ball next to the chair")])})
    with pytest.raises(ValueError, match="mask"):
        validate_row({**add, "edit_type": "remove"})
    umt = dict(sample_type="edit_umt", edit_type="add", edit_image="s.png", image="t.png",
               instr_variant="noref", prompt="Add a red ball in this region " + X + ".")
    assert validate_row(umt) is umt
    for prompt in ("Add a red ball in this region " + A + ".",
                   "Add a red ball in this region " + X + " " + Y + ".",
                   "Add a red ball in this region " + X + A + "."):
        with pytest.raises(ValueError):
            validate_row({**umt, "prompt": prompt})
    validate_inline("Add a red ball next to the chair " + X + Y + ".", "ref", "add")
    with pytest.raises(ValueError):
        validate_row({**umt, "edit_type": "composite"})


def test_rec_replay_rows():
    row = dict(sample_type="rec_ntp", edit_type="remove", edit_image="s.png",
               prompt=REC_TEMPLATE.format("red car"), mt_cot=to_cot([(X, "red car")]))
    assert validate_row(row) is row
    for change in (dict(prompt=REC_TEMPLATE.format("blue car")),
                   dict(mt_cot=to_cot([(A, "red car")])),
                   dict(mt_cot=to_cot([(X, "red car"), (Y, "car")])),
                   dict(prompt=REC_TEMPLATE.format("red car") + " " + X),
                   dict(image="t.png")):
        with pytest.raises(ValueError):
            validate_row({**row, **change})


@pytest.mark.parametrize(
    "stage,world,acc",
    [
        ("stage1", 1, 8),
        ("stage1", 2, 4),
        ("stage1", 8, 1),
        ("stage2", 1, 4),
        ("stage2", 2, 2),
        ("stage2", 8, 1),
    ],
)
def test_global_optimizer_batch_ratios(stage, world, acc):
    rows = [
        dict(sample_type=k, edit_type="attribute", **v)
        for k, v in [
            ("edit_ntp", {}),
            ("edit_umt", {"instr_variant": "ref"}),
            ("edit_umt", {"instr_variant": "noref"}),
            ("edit", {}),
        ]
    ]
    if stage == "stage2":
        rows = rows[1:]
    schedule, report = make_schedule(rows, stage, world, acc, steps=3, seed=17)
    assert len(schedule) == world * acc * 3
    for offset in range(0, len(schedule), world * acc):
        assert (
            Counter(row_kind(rows[i]) for i in schedule[offset : offset + world * acc])
            == report["per_step"]
        )
    assert make_schedule(rows, stage, world, acc, steps=3, seed=17)[0] == schedule
    with pytest.raises(ValueError):
        make_schedule(rows[:-1], stage, world, acc, steps=1)


@pytest.mark.parametrize(
    "types,counts",
    [
        (["attribute", "global"], [1, 100]),
        (["attribute", "global", "background"], [1, 100, 2]),
        (["attribute", "global", "background"], [100, 1, 100]),
    ],
)
def test_plain_edit_individual_caps(types, counts):
    probs = capped_plain_weights(types, counts)
    assert sum(probs) == pytest.approx(1)
    assert all(
        p <= 0.15 + 1e-12 for t, p in zip(types, probs) if t in {"global", "background"}
    )
