from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import EXPERIMENT_ROOT, read_rows, write_json, write_rows
from .model import DEFAULT_QWEN, DEFAULT_SAMTOK


def main():
    parser = argparse.ArgumentParser(description="SAMTok + official Qwen-Image-2.1")
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("build-debug")
    build.add_argument("--output", default=EXPERIMENT_ROOT + "/data")
    base = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    build.add_argument("--crisp", default=base + "CrispEdit-2M")
    build.add_argument("--masks", default=base + "CrispEdit-2M-mask")
    build.add_argument(
        "--gres", default=base + "SAMTok_Training_Data/mask_generation_gres209k.json"
    )
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
    validate = subs.add_parser("validate")
    validate.add_argument("--metadata", required=True)
    validate.add_argument("--base-path", default=".")
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
    for name in ("train", "cache", "infer", "localize"):
        p = subs.add_parser(name)
        p.add_argument("--qwen", default=DEFAULT_QWEN)
        p.add_argument("--samtok", default=DEFAULT_SAMTOK)
        p.add_argument("--output", required=True)
        if name in {"train", "cache"}:
            p.add_argument("--metadata")
            p.add_argument("--base-path", default=EXPERIMENT_ROOT + "/data")
            p.add_argument("--max-pixels", type=int, default=1048576)
        if name == "train":
            p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
            p.add_argument("--cache")
            p.add_argument("--steps", type=int)
            p.add_argument("--accumulation", type=int, default=8)
            p.add_argument("--rank", type=int)
            p.add_argument("--dropout", type=float)
            p.add_argument("--lr", type=float)
            p.add_argument(
                "--lr-schedule", choices=["constant", "cosine"], default="constant"
            )
            p.add_argument("--warmup-steps", type=int, default=0)
            p.add_argument("--weight-decay", type=float)
            p.add_argument("--max-grad-norm", type=float, default=1.0)
            p.add_argument("--ntp-weight", type=float, default=0.05)
            p.add_argument("--fm-weight", type=float, default=1.0)
            p.add_argument("--seed", type=int, default=20260920)
            p.add_argument("--save-every", type=int, default=100)
            p.add_argument(
                "--init-adapter",
                dest="resume_adapter",
                help="Warm start adapter weights; optimizer starts fresh",
            )
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
                p.add_argument("--cot-file")
                p.add_argument("--mask", nargs="+")
                p.add_argument("--dit-adapter")
                p.add_argument("--steps", type=int, default=40)
                p.add_argument("--cfg", type=float, default=1.0)
                p.add_argument("--no-kv-cache", action="store_true")
            else:
                p.add_argument("--candidates", type=int, default=1)
                p.add_argument("--decode-masks", action="store_true")
    args = parser.parse_args()
    if args.command == "build-debug":
        from .prepare import build_debug

        build_debug(args)
    elif args.command == "regions":
        from PIL import Image
        from .codec import SamtokCodec
        from .regions import segment, save_candidates

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
        from .prepare import convert_record

        rows, reports = [], []
        for i, line in enumerate(Path(args.input).read_text().splitlines()):
            derived, errors = convert_record(json.loads(line))
            rows.extend(derived)
            reports.append(
                {
                    "source_index": i,
                    "rewrite_errors": errors,
                    "output_rows": len(derived),
                }
            )
        write_rows(args.output, rows)
        write_json(args.output + ".report.json", reports)
    elif args.command == "validate":
        from PIL import Image

        rows = read_rows(args.metadata)
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
    elif args.command in {"train", "cache"}:
        from .training import train, cache

        if args.command == "train":
            defaults = {
                "stage1": dict(rank=64, dropout=0.05, lr=4e-5, weight_decay=0.05),
                "stage2": dict(rank=32, dropout=0.0, lr=1e-4, weight_decay=0.01),
            }[args.stage]
            for key, value in defaults.items():
                if getattr(args, key) is None:
                    setattr(args, key, value)
            if args.stage == "stage1" and not args.metadata:
                parser.error("stage1 requires --metadata")
            if args.stage == "stage2" and not args.cache:
                parser.error("stage2 requires --cache")
            train(args)
        else:
            if not args.metadata:
                parser.error("cache requires --metadata")
            cache(args)
    else:
        inference(args)


def inference(args):
    import numpy as np
    import torch
    from PIL import Image
    from .model import load_pipeline, edit, localize
    from .training import load_adapter

    if args.command == "localize" and args.candidates < 1:
        raise ValueError("candidates must be positive")
    torch.manual_seed(args.seed)

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
        from .training import adapter_identity

        config = load_adapter(pipe.dit, args.dit_adapter)
        expected = config.get("conditioning_identity", {}).get("te_adapter")
        actual = adapter_identity(args.te_adapter)
        if (expected or {}).get("sha256") != (actual or {}).get("sha256"):
            raise ValueError(
                "DiT checkpoint was trained with a different TE adapter; use its cache identity"
            )
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
            )
            for _ in range(args.candidates)
        ]
        if args.decode_masks:
            from .codec import SamtokCodec
            from .regions import decode_localizations

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
        from .codec import SamtokCodec
        from .protocol import interactive_prompt

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
    image.save(out)
    write_json(
        out.with_suffix(".json"),
        {
            **report,
            "args": vars(args),
            "output_mode": image.mode,
            "output_size": image.size,
        },
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
