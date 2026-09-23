import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]


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
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            include_dirs=_cuda_include_dirs(),
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
