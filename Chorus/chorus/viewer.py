from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

from chorus.input import get_default_output_dir, infer_scene_name

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
        prog="chorus-viewer",
        description="Launch installed Mini Viewer on a source 3DGS scene and Chorus features.",
    )
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--precompile-gsplat", action="store_true")
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--feature-type", default="siglip2")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--backend", default="auto", choices=["auto", "gsplat", "torch"])
    parser.add_argument("--pca-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pca-method", default="torch", choices=["torch", "sklearn"])
    parser.add_argument("--pca-brightness", type=float, default=1.25)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-splats", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def _require_installed_miniviewer() -> None:
    if importlib.util.find_spec("run_viewer") is not None:
        return
    raise ModuleNotFoundError(
        "Mini Viewer is not installed. Install the optional viewer dependencies, "
        "for example: python -m pip install "
        "\"git+https://github.com/RunyiYang/Mini_Viewer.git@6c8e5c938844487319a92e19f952e76cd4eba847\""
    )


def _infer_python_prefix() -> Path:
    executable = Path(sys.executable).resolve()
    return executable.parent.parent if executable.parent.name == "bin" else executable.parent


def _prepend_env_paths(env: dict[str, str], key: str, paths: list[Path]) -> None:
    existing = [item for item in env.get(key, "").split(os.pathsep) if item]
    prepended = []
    for path in paths:
        value = str(path)
        if path.exists() and value not in prepended and value not in existing:
            prepended.append(value)
    if prepended:
        env[key] = os.pathsep.join(prepended + existing)


def _detect_torch_cuda_arch_list() -> str | None:
    try:
        import torch

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
            detected = _detect_torch_cuda_arch_list()
            if detected:
                env["TORCH_CUDA_ARCH_LIST"] = detected
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


def _feature_path(args: argparse.Namespace, scene_name: str) -> Path | None:
    if args.feature_path and args.feature_path.strip().lower() not in {"none", "null"}:
        return Path(args.feature_path).expanduser()
    if args.feature_path and args.feature_path.strip().lower() in {"none", "null"}:
        return None
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else get_default_output_dir()
    return output_dir / f"{scene_name}_lang_feat.pt"


def build_command(args: argparse.Namespace, extra_args: list[str]) -> tuple[list[str], Path | None]:
    if not args.input_root:
        raise ValueError("Mini Viewer launch requires --input-root unless --precompile-gsplat is used.")
    input_path = Path(args.input_root).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    scene_name = infer_scene_name(input_path, explicit_name=args.scene_name)
    feature_path = _feature_path(args, scene_name)
    if feature_path is not None and not feature_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Feature file does not exist: {feature_path}")

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
    if args.max_splats is not None:
        cmd.extend(["--max-splats", str(args.max_splats)])
    cmd.extend(extra_args)
    return cmd, feature_path


def _print_env_summary(summary: dict[str, str]) -> None:
    print("Mini Viewer CUDA build environment:")
    for key, value in summary.items():
        if value:
            print(f"  {key}={value}")


def main() -> None:
    args, extra_args = parse_args()
    if not args.dry_run and not args.precompile_gsplat:
        _require_installed_miniviewer()
    env, env_summary = build_subprocess_env()
    if args.precompile_gsplat:
        _print_env_summary(env_summary)
        if args.dry_run:
            print(shlex.join([sys.executable, "-c", "<gsplat precompile snippet>"]))
            return
        subprocess.run([sys.executable, "-c", _GSPLAT_PRECOMPILE_CODE], check=True, env=env)
        return
    cmd, feature_path = build_command(args, extra_args)
    print("Mini Viewer command:")
    print(shlex.join(cmd))
    if feature_path is not None:
        print(f"Feature path: {feature_path}")
    if args.dry_run:
        _print_env_summary(env_summary)
        return
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
