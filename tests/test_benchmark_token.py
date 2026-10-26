"""The benchmark v2 adapter: the token-guidance negative prompt and the teacher test's allowed region."""


def test_token_guidance_negative_prompt_drops_every_region_token():
    from samtok_edit21.evaluation.benchmark_token import strip_tokens
    tok = "<|mt_start|><|mt_0012|><|mt_0345|><|mt_end|>"
    prompt = f"Execute all numbered instructions.\nU1: Remove the selected region {tok}.\nU2: Change the cup {tok} to red."
    assert strip_tokens(prompt) == ("Execute all numbered instructions.\nU1: Remove the selected region.\n"
                                    "U2: Change the cup to red.")
    assert "<|mt_" not in strip_tokens(prompt) and strip_tokens("Make it blue.") == "Make it blue."


def test_teacher_allowed_region_is_the_union_of_masks_plus_grown_add_boxes(tmp_path):
    import numpy as np
    from PIL import Image

    from samtok_edit21.evaluation.benchmark_token import allowed_region, ground_truth_mask

    size = (40, 20)                                     # width, height
    first = np.zeros((20, 40), bool)
    first[2:4, 2:4] = True
    second = np.zeros((20, 40), bool)
    second[10:12, 20:24] = True
    path = tmp_path / "U1.png"
    Image.fromarray(first.astype(np.uint8) * 255).resize((80, 40), Image.NEAREST).save(path)
    targets = {"U1": {"operation": "remove", "mask": str(path), "box": [2, 2, 4, 4]},
               "U2": {"operation": "add", "mask": "unused", "box": [20, 10, 24, 12]}}
    assert ground_truth_mask(targets["U1"], size).tolist() == first.tolist()   # resized to the source, nearest
    masks = {"U1": first, "U2": second}
    region = allowed_region(targets, masks, size, add_expand=0.5)
    expected = first | second
    expected[9:13, 18:26] = True                       # the add box grown by half its size on every side
    assert region.tolist() == expected.tolist()
    assert allowed_region(targets, masks, size, add_expand=0).tolist() == (first | second).tolist()

