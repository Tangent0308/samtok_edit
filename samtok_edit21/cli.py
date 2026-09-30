from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from .schema.data import EXPERIMENT_ROOT, read_rows, write_json, write_rows, row_hash, file_hash
from .models.model import DEFAULT_QWEN, DEFAULT_SAMTOK


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "prepare-regions":
        from .region.supervision import main as region_main
        return region_main(argv[1:])
    if argv and argv[0] == "calibrate-attention":
        from .training_core.calibrate_attention import main as calibration_main
        return calibration_main(argv[1:])
    if argv and argv[0] in {"train", "cache"}:
        from .training_core.train import main as training_main
        return training_main(argv)
    parser = argparse.ArgumentParser(description="SAMTok + Qwen-Image-2.1")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("train", help="Delegate to the canonical DiffSynth training entry point")
    subs.add_parser("cache", help="Delegate to the canonical cache-v2 builder")
    subs.add_parser("prepare-regions", help="Prepare frozen region supervision for both training stages")
    subs.add_parser("calibrate-attention", help="Calibrate A against actual C gradients without optimizer updates")
    build = subs.add_parser("build-debug")
    build.add_argument("--output", default=EXPERIMENT_ROOT + "/data")
    base = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    build.add_argument("--crisp", default=base + "CrispEdit-2M")
    build.add_argument("--masks", default=base + "CrispEdit-2M-mask")
    build.add_argument("--gres", help="Qwen3-VL-SAMTok GRES/GRefCOCO conversation JSON")
    build.add_argument("--gres-mask-tokenizer-sha256", help="Encoder checksum recorded by the GRES source builder")
    build.add_argument(
        "--gres-images",
        default="/mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k",
    )
    build.add_argument("--per-type", type=int, default=2)
    build.add_argument("--gres-count", type=int, default=4)
    build.add_argument("--samtok", default=DEFAULT_SAMTOK)
    build.add_argument("--device", default="cuda")
    convert = subs.add_parser("convert")
    convert.add_argument(
        "--input", required=True, help="JSONL common records with units/mask_codes"
    )
    convert.add_argument("--output", required=True)
    convert.add_argument("--samtok", default=DEFAULT_SAMTOK)
    convert.add_argument("--mask-tokenizer-sha256", required=True,
                         help="Checksum recorded by the input mask encoder; must match the supplied SAMTok codec")
    validate = subs.add_parser("validate")
    validate.add_argument("--metadata", required=True)
    validate.add_argument("--base-path", default=".")
    validate.add_argument("--check-bindings", action="store_true")
    region = subs.add_parser(
        "regions", help="SAM2 point/box proposals; masks and SAMTok codes for selection"
    )
    region.add_argument("--image", required=True)
    region.add_argument(
        "--point",
        nargs=3,
        type=float,
        action="append",
        default=[],
        metavar=("X", "Y", "LABEL"),
    )
    region.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    region.add_argument("--output", required=True)
    region.add_argument("--samtok", default=DEFAULT_SAMTOK)
    region.add_argument("--device", default="cuda")
    for name in ("infer", "localize"):
        p = subs.add_parser(name)
        p.add_argument("--qwen", default=DEFAULT_QWEN)
        p.add_argument("--samtok", default=DEFAULT_SAMTOK)
        p.add_argument("--output", required=True)
        if name in {"cache", "infer", "localize"}:
            p.add_argument("--te-adapter")
        if name in {"infer", "localize"}:
            p.add_argument("--image", nargs="+", required=True)
            p.add_argument("--prompt", required=True)
            p.add_argument("--height", type=int, default=1024)
            p.add_argument("--width", type=int, default=1024)
            p.add_argument("--device", default="cuda")
            p.add_argument("--max-new-tokens", type=int, default=256)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--variant", choices=("ref", "noref"), default="ref")
            p.add_argument("--strict-noref", action="store_true")
            p.add_argument("--units-file", help="Reviewed atomic units JSON list, in localization group order")
            if name == "infer":
                p.add_argument(
                    "--mode",
                    choices=[
                        "online",
                        "oracle",
                        "inline",
                        "direct",
                        "interactive",
                        "stock",
                    ],
                    default="online",
                )
                p.add_argument("--benchmark-output", action="store_true")
                p.add_argument("--reference-image-index", type=int)
                p.add_argument("--cot-file")
                p.add_argument("--mask", nargs="+")
                p.add_argument("--dit-adapter")
                p.add_argument("--steps", type=int, default=40)
                p.add_argument("--cfg", type=float, default=1.0)
                p.add_argument("--no-kv-cache", action="store_true")
            else:
                p.add_argument("--candidates", type=int, default=1)
                p.add_argument("--decode-masks", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build-debug":
        from .annotation.prepare import build_debug

        build_debug(args)
    elif args.command == "regions":
        from PIL import Image
        from .models.codec import SamtokCodec
        from .region.selection import segment, save_candidates

        codec = SamtokCodec(
            str(Path(args.samtok) / "sam2.1_hiera_large.pt"),
            str(Path(args.samtok) / "mask_tokenizer_256x2.pth"),
            device=args.device,
        )
        with Image.open(args.image) as im:
            image = im.convert("RGBA")
        masks, scores = segment(codec, image, points=args.point, box=args.box)
        records, selected = save_candidates(codec, image, masks, scores, args.output)
        print(
            json.dumps(
                {"candidates": records, "selected_index": selected}, ensure_ascii=False
            )
        )
    elif args.command == "convert":
        from .annotation.prepare import convert_record

        checksum = file_hash(Path(args.samtok) / "mask_tokenizer_256x2.pth")
        if args.mask_tokenizer_sha256 != checksum:
            raise ValueError("Input mask tokenizer checksum mismatch: re-encode raw masks with the supplied codec")
        for path in (args.output, args.output + ".report.json", args.output + ".provenance.json"):
            if Path(path).exists():
                raise ValueError("Conversion outputs must be fresh paths")
        rows, reports, manifest = [], [], []
        for i, line in enumerate(Path(args.input).read_text().splitlines()):
            if not line.strip():
                continue
            record = line
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Each source record must be a JSON object")
                derived, errors = convert_record(record)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                derived, errors = [], {"record": str(exc)}
            rows.extend(derived)
            metadata = record if isinstance(record, dict) else {}
            manifest.append({"source_dataset": metadata.get("source_dataset", str(Path(args.input).resolve())),
                             "source_index": i, "record": record,
                             "derived_row_hashes": [row_hash(r) for r in derived],
                             "rewrite_errors": errors,
                             "geometry_qc": "checked" if derived and metadata.get("units") and all("mask_paths" in u for u in metadata["units"]) else "upstream_required",
                             **{k: metadata[k] for k in ("codec_iou", "qc_flag", "mask_path", "decoded_path") if k in metadata}})
            reports.append(
                {
                    "source_index": i,
                    "rewrite_errors": errors,
                    "output_rows": len(derived),
                }
            )
        write_rows(args.output, rows)
        write_json(args.output + ".provenance.json", manifest)
        write_json(args.output + ".report.json", {"mask_tokenizer_sha256": checksum,
                   "input_sha256": file_hash(args.input), "rows": len(rows), "records": reports})
    elif args.command == "validate":
        from PIL import Image

        rows = read_rows(args.metadata)
        if args.check_bindings:
            from .schema.protocol import grouped_units, parse_cot, render_units
            failures = []
            for index, row in enumerate(rows):
                if row["sample_type"] == "edit_ntp":
                    try:
                        render_units(row["prompt"], grouped_units(row["prompt"], parse_cot(row["mt_cot"])))
                    except ValueError as exc:
                        failures.append({"row": index, "reason": str(exc)})
            print(json.dumps({"binding_failures": failures, "ntp_rows": sum(r["sample_type"] == "edit_ntp" for r in rows)}))
            if failures:
                raise SystemExit(1)
        paths = set()
        for row in rows:
            sources = row["edit_image"]
            sources = [sources] if isinstance(sources, str) else sources
            paths.update(sources)
            if "image" in row:
                paths.add(row["image"])
        for path in sorted(paths):
            with Image.open(Path(args.base_path) / path) as im:
                im.load()
        print(
            json.dumps(
                {"rows": len(rows), "decoded_images": len(paths), "passed": True}
            )
        )
    else:
        inference(args)


def inference(args):
    import numpy as np
    import torch
    from PIL import Image
    from .models.model import load_pipeline, edit, localize
    from .training_core.training import load_adapter

    if args.command == "localize" and args.candidates < 1:
        raise ValueError("candidates must be positive")
    from accelerate.utils import set_seed
    from .schema.protocol import spans_in
    set_seed(args.seed)
    masks = spans_in(args.prompt)
    mode = getattr(args, "mode", None)
    if (args.command == "localize" or mode in {"online", "oracle", "interactive"} or masks) and len(args.image) != 1:
        raise ValueError("Mask/localization modes require exactly one source image")
    if mode in {"direct", "stock"} and masks:
        raise ValueError("direct/stock are plain modes; masks require inline")
    if mode == "inline" and not masks:
        raise ValueError("inline requires mask spans")
    if mode == "oracle" and not args.cot_file:
        raise ValueError("oracle requires --cot-file")
    if mode == "interactive" and not args.mask:
        raise ValueError("interactive requires --mask")
    if args.strict_noref and args.variant != "noref":
        raise ValueError("--strict-noref requires --variant noref")
    if args.strict_noref and args.command == "infer" and mode not in {"online", "oracle"}:
        raise ValueError("--strict-noref is only defined for online/oracle")
    if getattr(args, "benchmark_output", False):
        if len(args.image) > 1 and args.reference_image_index is None:
            raise ValueError("Multi-image benchmark requires --reference-image-index")
        reference = args.reference_image_index if args.reference_image_index is not None else 0
        if not 0 <= reference < len(args.image):
            raise ValueError("Invalid reference image index")
    reviewed = json.loads(Path(args.units_file).read_text()) if args.units_file else None
    if getattr(args, "dit_adapter", None):
        from .schema.provenance import assert_inference_identity
        config = json.loads((Path(args.dit_adapter) / "adapter.json").read_text())
        if config["stage"] != "stage2":
            raise ValueError("--dit-adapter must belong to Stage 2")
        assert_inference_identity(config.get("conditioning_identity", {}),
                                  args.qwen, args.samtok, args.te_adapter)

    if args.command == "infer" and Path(args.output).suffix.lower() != ".png":
        raise ValueError("Save RGBA output as .png")
    images = [Image.open(p).convert("RGBA") for p in args.image]
    stock = getattr(args, "mode", None) == "stock"
    if stock and (args.te_adapter or args.dit_adapter):
        raise ValueError("Stock baseline must not load project adapters")
    pipe = load_pipeline(
        args.qwen,
        None if stock else args.samtok,
        device=args.device,
        components=("text_encoder",)
        if args.command == "localize"
        else ("text_encoder", "dit", "vae"),
    )
    if args.te_adapter:
        load_adapter(pipe.text_encoder, args.te_adapter)
    if getattr(args, "dit_adapter", None):
        load_adapter(pipe.dit, args.dit_adapter)
    pipe.eval()
    if args.command == "localize":
        results = [
            localize(
                pipe,
                args.prompt,
                images,
                height=args.height,
                width=args.width,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.candidates > 1,
                variant=args.variant, reviewed_units=reviewed, strict_noref=args.strict_noref,
            )
            for _ in range(args.candidates)
        ]
        if args.decode_masks:
            from .models.codec import SamtokCodec
            from .region.selection import decode_localizations

            codec = SamtokCodec(
                str(Path(args.samtok) / "sam2.1_hiera_large.pt"),
                str(Path(args.samtok) / "mask_tokenizer_256x2.pth"),
                device=args.device,
            )
            decode_localizations(
                codec, images[0], results, Path(args.output).with_suffix("")
            )
        write_json(args.output, results)
        # Each sample is an alternative hypothesis. Items within one sample can
        # mean multiple simultaneous target instances, not alternative choices.
        print(json.dumps(results, ensure_ascii=False))
        return
    prompt = args.prompt
    mode = "direct" if stock else args.mode
    if mode == "interactive":
        from .models.codec import SamtokCodec
        from .schema.protocol import interactive_prompt

        if len(images) != 1 or not args.mask:
            raise ValueError("Interactive mode requires one source image and --mask")
        codec = SamtokCodec(
            str(Path(args.samtok) / "sam2.1_hiera_large.pt"),
            str(Path(args.samtok) / "mask_tokenizer_256x2.pth"),
            device=args.device,
        )
        masks = [np.asarray(Image.open(p).convert("L")) > 0 for p in args.mask]
        # Preserve selected-region order; codec sorting is applied only inside a group.
        groups = [codec.encode(images[0], [m])[0] for m in masks]
        prompt = interactive_prompt(
            prompt, groups, whole_image=len(masks) == 1 and bool(masks[0].all())
        )
        del codec
        mode = "inline"
    cot = Path(args.cot_file).read_text() if args.cot_file else None
    image, report = edit(
        pipe,
        prompt,
        images,
        mode=mode,
        cot=cot,
        variant=args.variant, reviewed_units=reviewed, strict_noref=args.strict_noref,
        max_new_tokens=args.max_new_tokens,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        use_kv_cache=not args.no_kv_cache,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".png":
        raise ValueError("Save RGBA output as .png")
    raw_size = image.size
    raw_path = None
    if args.benchmark_output:
        raw_path = out.with_name(out.stem + ".raw.png")
        image.save(raw_path)
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
        image = image.resize(images[reference].size, Image.Resampling.LANCZOS)
    image.save(out)
    write_json(
        out.with_suffix(".json"),
        {
            **report,
            "args": vars(args),
            "output_mode": image.mode,
            "output_size": image.size,
            "raw_size": raw_size, "raw_output": str(raw_path) if raw_path else None,
            "postprocessing": "white-alpha-composite+reference-resize" if args.benchmark_output else "native",
        },
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
