from setuptools import find_packages, setup


setup(
    name="chorus",
    version="0.1.0",
    description="Inference-only Chorus 3DGS encoder package",
    packages=find_packages(include=["chorus", "chorus.*"]),
    include_package_data=True,
    install_requires=[
        "addict",
        "huggingface_hub",
        "numpy",
        "plyfile",
        "scipy",
        "timm",
    ],
    entry_points={
        "console_scripts": [
            "chorus-encode=chorus.cli:main",
            "chorus-viewer=chorus.viewer:main",
        ],
    },
)
