from __future__ import annotations

import argparse
import json
from pathlib import Path

from chorus import load
from chorus.input import get_default_output_dir


def _parse_outputs(value: str):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_json_file(path: str | None):
    if path is None:
        return None
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chorus-encode",
        description="Run detached package-mode Chorus feature encoding.",
    )
    parser.add_argument("--mode", default="chorus_3dgs", choices=["chorus_3dgs", "chorus_pts"])
    parser.add_argument("--checkpoint", default=None, help="Local checkpoint path or HF filename/stem.")
    parser.add_argument("--input-root", required=True, help="Scene folder or raw .ply input.")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--output-dir", default=None, help="Defaults to ./outputs for CLI saves.")
    parser.add_argument(
        "--outputs",
        default="lang",
        help="Comma-separated outputs: lang,dino,backbone_upcast,backbone_last.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--return-torch", action="store_true", help="Keep returned features as torch tensors internally.")
    parser.add_argument("--no-save", action="store_true", help="Run encoding without writing output files.")
    parser.add_argument("--disable-outlier-filter", action="store_true")
    parser.add_argument(
        "--outlier-filter-json",
        default=None,
        help="Optional JSON object file with raw-Ply outlier filter overrides.",
    )
    parser.add_argument("--summary-json", default=None, help="Optional path to write a JSON run summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outlier_filter = _parse_json_file(args.outlier_filter_json)
    if args.disable_outlier_filter:
        outlier_filter = False if outlier_filter is None else {**outlier_filter, "enabled": False}
    output_dir = None if args.no_save else Path(args.output_dir).expanduser() if args.output_dir else get_default_output_dir()

    encoder = load(
        args.mode,
        checkpoint=args.checkpoint,
        outputs=_parse_outputs(args.outputs),
        device=args.device,
        return_numpy=not args.return_torch,
        chunk_size=args.chunk_size,
        output_dir=output_dir,
        outlier_filter=outlier_filter,
    )
    output = encoder.encode(
        args.input_root,
        scene_name=args.scene_name,
        save=not args.no_save,
    )
    summary = {
        "name": output.name,
        "mode": args.mode,
        "outputs": list(output.features.keys()) + list(output.tokens.keys()),
        "feature_shapes": {
            name: list(value.shape) for name, value in output.features.items()
        },
        "token_shapes": {
            name: list(token.feat.shape) for name, token in output.tokens.items()
        },
        "metadata": {
            key: value
            for key, value in output.metadata.items()
            if key
            in {
                "input_path",
                "source_type",
                "source_raw_count",
                "source_kept_count",
                "outlier_filter_enabled",
            }
        },
        "output_dir": str(output_dir) if output_dir else None,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
