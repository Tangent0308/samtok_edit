"""Keep model-visible requests and offline scoring truth in separate files."""

from adapter import adapt
from common import ROOT, read_jsonl, sha256, write_json, write_jsonl

from samtok_edit21.pass1.protocol import prompt


def main():
    cases = [
        r for p in sorted((ROOT / "data").glob("codec_*.jsonl")) for r in read_jsonl(p)
    ]
    assert len(cases) == len({x["id"] for x in cases}) == 20
    by_id = {x["id"]: x for x in cases}
    tokens = {
        (c["image_sha256"], u["mask_sha256"]): u["span"]
        for c in cases
        for u in c["units"]
    }
    public_rows, scoring_rows = [], []
    for public in read_jsonl(ROOT / "data/benchmark_public.jsonl"):
        c = by_id[public["case_id"]]
        request = adapt(public, tokens)
        row = {
            "id": public["request_id"],
            "group": c["group"],
            "image": public["source"]["path"],
            "image_sha256": public["source"]["sha256"],
            "size": public["source"]["size"],
            "request": request,
        }
        public_rows.append(row)
        truth = [
            dict(u, family="IF-Bench-2.6.0", record=c["id"], original_uid=u["label"])
            for u in c["units"]
        ]
        scoring_rows.append(
            dict(
                row,
                truth=truth,
                answer=[{"label": u["label"], "mask_2d": u["span"]} for u in truth],
            )
        )
    write_jsonl(ROOT / "data/inference_public.jsonl", public_rows)
    write_jsonl(ROOT / "data/scoring_truth.jsonl", scoring_rows)
    write_json(
        ROOT / "provenance/inputs.json",
        {
            "requests": len(public_rows),
            "inference_sha256": sha256(ROOT / "data/inference_public.jsonl"),
            "scoring_sha256": sha256(ROOT / "data/scoring_truth.jsonl"),
            "private_scoring_input_passed_to_generate": False,
        },
    )
    for setting in ["S1", "S2-point", "S3-box", "S5"]:
        row = next(x for x in public_rows if x["id"].endswith("--" + setting))
        (ROOT / "provenance" / f"prompt_{setting}.txt").write_text(
            prompt(row["request"])
        )
    print("Prepared 200 public requests and separate offline truth.", flush=True)


if __name__ == "__main__":
    main()
