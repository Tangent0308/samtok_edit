"""Region-token protocol: mask spans, boxes and their validation."""
import pytest

from samtok_edit21.data.protocol import box_coords, box_of, boxes_in, regions_in, span_of, spans_in

A = span_of([3, 300])
X, Y = box_of((100, 200, 300, 400)), box_of((0, 0, 1000, 1000))


def test_two_codebooks():
    assert A == "<|mt_start|><|mt_0003|><|mt_0300|><|mt_end|>"
    for codes in ([256, 300], [10, 255], [1, 512], [1]):
        with pytest.raises(ValueError):
            span_of(codes)
    with pytest.raises(ValueError):
        spans_in("<|mt_start|><|mt_0001|><|mt_end|>")
    with pytest.raises(ValueError):
        spans_in("<|mt_start|><|mt_0300|><|mt_0003|><|mt_end|>")     # level-2 code in the level-1 slot
    assert spans_in("Remove the selected region " + A + ".") == [A]


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
