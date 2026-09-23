import os
import sys
from sys import argv
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)


def _argparse(pattern, argv, is_flag=True, is_list=False):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                val = arr[0].split("=")[1]
                if "," in val:
                    return val.split(","), argv
                else:
                    return [val], argv
        else:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                return arr[0].split("=")[1], argv


INCLUDE_DIRS, argv = _argparse("--include_dirs", argv, False, is_list=True)
include_dirs = []
if INCLUDE_DIRS is not False:
    include_dirs += INCLUDE_DIRS


def _cuda_include_dirs():
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
    ]
    cuda_runtime = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "nvidia",
        "cuda_runtime",
        "include",
    )
    candidates = []
    for root in roots:
        if root:
            candidates.extend(
                [
                    os.path.join(root, "include"),
                    os.path.join(root, "targets", "x86_64-linux", "include"),
                    os.path.join(root, "targets", "x86_64-linux", "include", "cccl"),
                ]
            )
    candidates.append(cuda_runtime)
    return [
        path
        for index, path in enumerate(candidates)
        if path and os.path.isdir(path) and path not in candidates[:index]
    ]

setup(
    name="pointgroup_ops",
    packages=["pointgroup_ops"],
    package_dir={"pointgroup_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointgroup_ops_cuda",
            sources=["src/bfs_cluster.cpp", "src/bfs_cluster_kernel.cu"],
            include_dirs=[*include_dirs, *_cuda_include_dirs()],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
