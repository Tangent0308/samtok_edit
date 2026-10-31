# Pass1 experiment procedures

Reusable implementation is in `src/samtok_edit21/pass1/`; these scripts perform source audits and reproduce specific experiment reports. They do not load models or write results merely by being imported, except that the fixed dev20 helpers require explicit environment configuration when imported.

| Directory | Purpose | Inputs / output configuration |
|---|---|---|
| `data/` | Inventory, independent image-source audit, audited selection | Explicit CLI paths; see each script's `--help` |
| `formal/` | Status, save/resume checks, complete formal-run report | `PASS1_FORMAL_ROOT`; report images also require `PASS1_PREVIEW_ROOT` |
| `ifbench_dev20/` | Frozen official dev20 preparation, joint inference, scoring, gallery and failure analysis | Required `PASS1_BENCH_ROOT`, `PASS1_FORMAL_ROOT`, `PASS1_BENCH_REPO`, `PASS1_BENCH_DATASET`, `PASS1_DEV20_BUNDLE`, `PASS1_PREVIEW_ROOT`; optional `PASS1_ADAPTER` |

The dev20 suite deliberately checks the frozen 20-image/72-target study and step2000 training exposure. It is not an arbitrary test-set runner. Its report scripts operate on a new output directory or a completed run deliberately selected by the caller. Keep original experiment artifacts immutable.

`check_resume_equivalence.py` is the historical two-step numerical replay diagnostic. Its strict equality criterion did **not** pass in the recorded GPU experiment; this is separate from successful FP32 adapter loading, checkpoint-state restoration and adapter-off TE equality. Do not present it as a passing formal-run requirement.

Run commands, environment constraints, original run locations and the limits of the measurements are documented in [the reproduction guide](../../docs/pass1_reproduction.md).
