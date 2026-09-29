import pytest

from samtok_edit21.annotate_full import parse_output


def test_reasoning_is_not_parsed_as_annotation():
    answer = '{"ref_phrase": ["red cup"], "noref_instruction": "Replace this region."}'
    assert parse_output(answer)['ref_phrase'] == ['red cup']
    # Qwen3.5's chat template places <think> in the prompt, so generated text
    # starts directly with reasoning and contains only the closing marker.
    assert parse_output('Reasoning text that is not JSON.</think>' + answer) == parse_output(answer)
    assert parse_output('{"ref_phrase": ["sign"], "noref_instruction": "Write </think> here."}')['noref_instruction'] == 'Write </think> here.'
    with pytest.raises(ValueError):
        parse_output('Reasoning text without a closing marker or answer')
