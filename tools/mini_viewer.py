from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-512"

_GSPLAT_PRECOMPILE_CODE = r"""
import torch
from gsplat.rendering import rasterization

means = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
means[:, 2] = 1.0
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
scales = torch.ones((1, 3), device="cuda") * 0.01
opacities = torch.ones((1,), device="cuda")
colors = torch.ones((1, 3), device="cuda")
viewmats = torch.eye(4, device="cuda")[None]
Ks = torch.tensor([[[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]], device="cuda")
render_colors, _, _ = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=32,
    height=32,
    packed=False,
)
torch.cuda.synchronize()
print("gsplat precompile OK", tuple(render_colors.shape))
"""


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="python -m tools.mini_viewer",
        description=(
            "Launch installed Mini Viewer on a source 3DGS scene and saved "
            "SceneSplat inference features."
        ),
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Original scene directory with .npy files, or the raw .ply used for inference.",
    )
    parser.add_argument(
        "--precompile-gsplat",
        action="store_true",
        help=(
            "Run one tiny gsplat CUDA rasterization to JIT-build gsplat_cuda, "
            "then exit. Use this once before the first CUDA/gsplat viewer run."
        ),
    )
    parser.add_argument(
        "--feature-path",
        default=None,
        help=(
            "Feature tensor to visualize. Defaults to "
            "<output-dir>/<scene>_feat.pt, matching SceneSplat inference. "
            "Pass 'none' to view only the original splats."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Inference output directory. Defaults to <repo_root>/outputs.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name override.")
    parser.add_argument(
        "--feature-type",
        default="siglip2",
        choices=["siglip2", "siglip", "clip", "dino", "dinov2", "dino2"],
        help="Mini Viewer feature/query family. SceneSplat language features use siglip2.",
    )
    parser.add_argument("--query-feature", default=None, help="Optional precomputed query vector.")
    parser.add_argument("--query-image", default=None, help="Optional initial DINO/DINOv2 image query.")
    parser.add_argument(
        "--siglip-model",
        default=_DEFAULT_SIGLIP_MODEL,
        help="SigLIP/SigLIP2 text-query model id. Defaults to the 768-D SceneSplat model.",
    )
    parser.add_argument("--dino-model", default=None, help="Optional DINO/DINOv2 model id.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--enable-feature-model-on-cpu",
        action="store_true",
        help="Allow Mini Viewer text/image query encoders on CPU.",
    )
    parser.add_argument("--bbox-script", default=None, help="Optional Mini Viewer bbox script.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--backend", default="auto", choices=["auto", "gsplat", "torch"])
    parser.add_argument("--pca-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pca-method", default="torch", choices=["torch", "sklearn"])
    parser.add_argument("--pca-brightness", type=float, default=1.25)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-splats", type=int, default=None)
    parser.add_argument("--max-cpu-splats", type=int, default=None)
    parser.add_argument("--cpu-fallback-splats", type=int, default=None)
    parser.add_argument("--no-cpu-render-fallback", action="store_true")
    parser.add_argument("--force-cpu-render", action="store_true")
    parser.add_argument("--npy-scale-log", action="store_true")
    parser.add_argument("--camera-path", default=None)
    parser.add_argument("--video-output", default=None)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=None)
    parser.add_argument("--render-fps", type=int, default=None)
    parser.add_argument("--render-seconds", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved Mini Viewer command without launching the server.",
    )
    return parser.parse_known_args()


def _require_installed_miniviewer() -> None:
    if importlib.util.find_spec("run_viewer") is not None:
        return
    raise ModuleNotFoundError(
        "Mini Viewer is not installed in the active Python environment. "
        "Install it with: python -m pip install \"nerfview>=0.1.3\" "
        "\"splines>=0.3\" "
        "\"git+https://github.com/RunyiYang/Mini_Viewer.git@"
        "6c8e5c938844487319a92e19f952e76cd4eba847\""
    )


def _infer_scene_name(input_path: str | Path, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    path = Path(input_path).expanduser()
    if path.is_file() and path.suffix:
        return path.stem
    return path.name


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    return _REPO_ROOT / "outputs"


def _resolve_feature_path(
    *,
    feature_path: str | None,
    output_dir: Path,
    scene_name: str,
) -> Path | None:
    if feature_path is None:
        return output_dir / f"{scene_name}_feat.pt"
    if str(feature_path).strip().lower() in {"", "none", "null"}:
        return None
    return Path(feature_path).expanduser()


def _append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _prepend_env_paths(env: dict[str, str], key: str, paths: list[Path]) -> None:
    existing = [item for item in env.get(key, "").split(os.pathsep) if item]
    prepended = []
    for path in paths:
        value = str(path)
        if path.exists() and value not in prepended and value not in existing:
            prepended.append(value)
    if prepended:
        env[key] = os.pathsep.join(prepended + existing)


def _infer_python_prefix() -> Path:
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "bin":
        return executable.parent.parent
    return executable.parent


def _detect_torch_cuda_arch_list() -> str | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        archs = []
        for device_idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_idx)
            arch = f"{major}.{minor}"
            if arch not in archs:
                archs.append(arch)
        return ";".join(archs) if archs else None
    except Exception:
        return None


def build_subprocess_env() -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    prefix = _infer_python_prefix()
    nvcc = prefix / "bin" / "nvcc"
    target_include = prefix / "targets" / "x86_64-linux" / "include"
    cccl_include = target_include / "cccl"

    summary = {
        "python_prefix": str(prefix),
        "CUDA_HOME": env.get("CUDA_HOME", ""),
        "CONDA_PREFIX": env.get("CONDA_PREFIX", ""),
        "target_include": str(target_include) if target_include.exists() else "",
        "cccl_include": str(cccl_include) if cccl_include.exists() else "",
        "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", ""),
    }

    if nvcc.exists():
        env["CUDA_HOME"] = str(prefix)
        env["CONDA_PREFIX"] = str(prefix)
        _prepend_env_paths(env, "PATH", [prefix / "bin"])
        _prepend_env_paths(env, "CPATH", [target_include, cccl_include])
        _prepend_env_paths(env, "CPLUS_INCLUDE_PATH", [target_include, cccl_include])
        if not env.get("TORCH_CUDA_ARCH_LIST"):
            detected_archs = _detect_torch_cuda_arch_list()
            if detected_archs:
                env["TORCH_CUDA_ARCH_LIST"] = detected_archs

        summary.update(
            {
                "CUDA_HOME": env.get("CUDA_HOME", ""),
                "CONDA_PREFIX": env.get("CONDA_PREFIX", ""),
                "CPATH": env.get("CPATH", ""),
                "CPLUS_INCLUDE_PATH": env.get("CPLUS_INCLUDE_PATH", ""),
                "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", ""),
            }
        )
    return env, summary


def build_command(args: argparse.Namespace, extra_args: list[str]) -> tuple[list[str], Path | None]:
    if not args.input_root:
        raise ValueError("Mini Viewer launch requires --input-root unless --precompile-gsplat is used.")
    input_path = Path(args.input_root).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    scene_name = _infer_scene_name(input_path, explicit_name=args.scene_name)
    output_dir = _resolve_output_dir(args.output_dir)
    feature_path = _resolve_feature_path(
        feature_path=args.feature_path,
        output_dir=output_dir,
        scene_name=scene_name,
    )

    if feature_path is not None and not feature_path.exists() and not args.dry_run:
        raise FileNotFoundError(
            "Mini Viewer feature file does not exist: "
            f"{feature_path}. Run SceneSplat inference first or pass --feature-path."
        )

    cmd = [sys.executable, "-m", "run_viewer"]
    if input_path.is_dir():
        cmd.extend(["--folder-npy", str(input_path)])
    elif input_path.is_file() and input_path.suffix.lower() == ".ply":
        cmd.extend(["--ply", str(input_path)])
    else:
        raise ValueError(f"Unsupported input path for Mini Viewer: {input_path}")

    if feature_path is not None:
        cmd.extend(["--feature-file", str(feature_path)])

    cmd.extend(
        [
            "--feature-type",
            args.feature_type,
            "--device",
            args.device,
            "--backend",
            args.backend,
            "--pca-device",
            args.pca_device,
            "--pca-method",
            args.pca_method,
            "--pca-brightness",
            str(args.pca_brightness),
            "--pca-seed",
            str(args.pca_seed),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )

    _append_optional(cmd, "--query-feature", args.query_feature)
    _append_optional(cmd, "--query-image", args.query_image)
    _append_optional(cmd, "--siglip-model", args.siglip_model)
    _append_optional(cmd, "--dino-model", args.dino_model)
    _append_optional(cmd, "--hf-cache-dir", args.hf_cache_dir)
    _append_optional(cmd, "--bbox-script", args.bbox_script)
    _append_optional(cmd, "--max-splats", args.max_splats)
    _append_optional(cmd, "--max-cpu-splats", args.max_cpu_splats)
    _append_optional(cmd, "--cpu-fallback-splats", args.cpu_fallback_splats)
    _append_optional(cmd, "--camera-path", args.camera_path)
    _append_optional(cmd, "--video-output", args.video_output)
    _append_optional(cmd, "--render-width", args.render_width)
    _append_optional(cmd, "--render-height", args.render_height)
    _append_optional(cmd, "--render-fps", args.render_fps)
    _append_optional(cmd, "--render-seconds", args.render_seconds)

    if args.enable_feature_model_on_cpu:
        cmd.append("--enable-feature-model-on-cpu")
    if args.no_cpu_render_fallback:
        cmd.append("--no-cpu-render-fallback")
    if args.force_cpu_render:
        cmd.append("--force-cpu-render")
    if args.npy_scale_log:
        cmd.append("--npy-scale-log")
    cmd.extend(extra_args)
    return cmd, feature_path


def _print_cuda_build_env(env_summary: dict[str, str]) -> None:
    print("Mini Viewer CUDA build environment:")
    for key in (
        "python_prefix",
        "CUDA_HOME",
        "CONDA_PREFIX",
        "target_include",
        "cccl_include",
        "TORCH_CUDA_ARCH_LIST",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
    ):
        value = env_summary.get(key, "")
        if value:
            print(f"  {key}={value}")


def precompile_gsplat(env: dict[str, str]) -> None:
    print("Precompiling gsplat CUDA extension for Mini Viewer.")
    print("This is a one-time JIT build and can take several minutes on a fresh environment.")
    subprocess.run([sys.executable, "-c", _GSPLAT_PRECOMPILE_CODE], check=True, env=env)


def main() -> None:
    args, extra_args = parse_args()
    _require_installed_miniviewer()
    env, env_summary = build_subprocess_env()

    if args.precompile_gsplat:
        _print_cuda_build_env(env_summary)
        if args.dry_run:
            print("Would precompile gsplat with:")
            print(shlex.join([sys.executable, "-c", "<gsplat precompile snippet>"]))
            return
        precompile_gsplat(env)
        return

    if not args.input_root:
        raise SystemExit("--input-root is required unless --precompile-gsplat is used.")

    cmd, feature_path = build_command(args, extra_args)
    print("Mini Viewer command:")
    print(shlex.join(cmd))
    if feature_path is not None:
        index_path = feature_path.with_name(f"{feature_path.stem}_index.npy")
        print(f"Feature path: {feature_path}")
        print(f"Feature index sidecar: {index_path if index_path.exists() else 'not found'}")
    if args.dry_run:
        _print_cuda_build_env(env_summary)
        return
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
