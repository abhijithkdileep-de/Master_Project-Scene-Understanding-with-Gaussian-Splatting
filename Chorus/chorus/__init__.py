"""Inference-only Chorus package API."""

__version__ = "0.1.0"


def load(*args, **kwargs):
    """Load a Chorus encoder preset.

    Imports the CUDA-heavy runtime lazily so `import chorus` stays lightweight.
    """

    from .api import load as _load

    return _load(*args, **kwargs)


__all__ = ["load", "__version__"]
