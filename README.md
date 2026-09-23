# SAMTokEdit with the official Qwen-Image-2.1 training interfaces

This branch starts from `samtok_edit/main`, vendors DiffSynth 2.1.8, and adapts the SAMTok localization-then-edit method to the official Qwen-Image-2.1 pipeline.

The main entry point is `samtok_edit21.official_api`. It uses DiffSynth's official `launch_data_process_task` and `launch_training_task`, with a SAMTok text encoder, NTP/FM Stage 1 loss, deterministic type-ratio schedules, and cache integrity checks.

```bash
source /opt/tiger/tanyue/samtok_edit_qwen_image_2_1/.venv/bin/activate
PYTHONPATH=.:DiffSynth-Studio python -m pytest -q
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.official_api --help
```

The detailed implementation and experiment records are in:

- [SAMTokEdit_Qwen21_官方接口扩展实现.md](SAMTokEdit_Qwen21_官方接口扩展实现.md)
- [SAMTokEdit_Qwen21_官方接口实验记录.md](SAMTokEdit_Qwen21_官方接口实验记录.md)

The source dataset and model directories are read-only inputs. Smoke metadata is under `smoke_data/` and only references the supplied source images.
