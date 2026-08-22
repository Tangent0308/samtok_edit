"""State-dict conversion for released Qwen2.5-VL SAMTok checkpoints."""


def QwenImageSamtokTextEncoderStateDictConverter(state_dict):
    """Normalize both old and new HF Qwen2.5-VL export layouts.

    ``state_dict`` may be DiffSynth's disk-backed mapping, so this function only
    relies on key iteration and indexed reads.
    """

    converted = {}
    for key in state_dict:
        value = state_dict[key]
        if key.startswith("visual."):
            key = "model." + key
        elif key.startswith("model.language_model.") or key.startswith("model.visual."):
            pass
        elif key.startswith("model."):
            key = key.replace("model.", "model.language_model.", 1)
        converted[key] = value
    embedding_key = "model.language_model.embed_tokens.weight"
    if "lm_head.weight" not in converted and embedding_key in converted:
        converted["lm_head.weight"] = converted[embedding_key]
    return converted
