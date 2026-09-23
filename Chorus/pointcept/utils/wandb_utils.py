""" to guard wandb logging during distributed training."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - wandb is optional at runtime
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

from pointcept.utils import comm

_LOGGER = logging.getLogger("pointcept.wandb")


def is_wandb_active() -> bool:
    """Return True when a W&B run is available on the current process."""
    if wandb is None:
        return False
    if not comm.is_main_process():
        return False
    run = getattr(wandb, "run", None)
    return run is not None


def safe_wandb_log(
    data: Dict[str, Any],
    *,
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Safely forward metrics to W&B when a run is active.

    On non-main processes or when no run exists, the call becomes a no-op. Any
    backend exceptions are caught and only logged locally so that training does
    not crash because of logging.
    """

    if not is_wandb_active():
        return

    run_step = getattr(wandb.run, "step", None)
    log_kwargs: Dict[str, Any] = {}
    if step is not None:
        log_kwargs["step"] = step
    elif run_step is not None:
        log_kwargs["step"] = run_step
    if commit is not None:
        log_kwargs["commit"] = commit

    try:
        wandb.log(data, **log_kwargs)
    except Exception as exc:  # pragma: no cover - best effort safeguard
        target_logger = logger if logger is not None else _LOGGER
        target_logger.warning("Skipping wandb log due to error: %s", exc)


def safe_wandb_define_metric(*args: Any, **kwargs: Any) -> None:
    """Define a wandb metric if a run is active."""
    if not is_wandb_active():
        return
    try:
        wandb.define_metric(*args, **kwargs)
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("Failed to define wandb metric: %s", exc)
