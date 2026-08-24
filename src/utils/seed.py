"""Global seeding and RNG-state capture/restore for exact training resume.

``seed_all(seed)`` is the single entry point required by CONTEXT.md 7
("deterministic given ``src/utils/seed.py::seed_all(seed)``").  RNG state
helpers back the per-epoch checkpoints of CONTEXT.md 10.

Torch is imported lazily so that config/IO utilities remain importable before
the (large) torch wheel finishes installing.
"""

from __future__ import annotations

import os
import random
from typing import Any

__all__ = ["seed_all", "get_rng_state", "set_rng_state", "worker_init_fn"]


def _maybe_torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # pragma: no cover - torch absent during bootstrap
        return None


def _maybe_numpy():
    try:
        import numpy  # noqa: PLC0415

        return numpy
    except Exception:  # pragma: no cover
        return None


def seed_all(seed: int, *, deterministic: bool = True) -> int:
    """Seed python / numpy / torch (CPU + all CUDA devices).

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic:
        When True (default) also sets ``cudnn.deterministic = True`` and
        ``cudnn.benchmark = False``.  We do *not* call
        ``torch.use_deterministic_algorithms`` because scatter-add
        (``index_add_``, used by the GNN) has no deterministic CUDA kernel and
        would raise.

    Returns
    -------
    int
        The seed, for convenient logging.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np = _maybe_numpy()
    if np is not None:
        np.random.seed(seed % (2**32))

    torch = _maybe_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:  # pragma: no cover - no cudnn build
                pass
    return seed


def get_rng_state() -> dict[str, Any]:
    """Capture python/numpy/torch RNG states for a checkpoint."""
    state: dict[str, Any] = {"python": random.getstate()}

    np = _maybe_numpy()
    if np is not None:
        state["numpy"] = np.random.get_state()

    torch = _maybe_torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG states produced by :func:`get_rng_state` (missing keys skipped)."""
    if not state:
        return
    if "python" in state:
        pystate = state["python"]
        # torch.save round-trips tuples as lists in some versions.
        if isinstance(pystate, list):
            pystate = (pystate[0], tuple(pystate[1]), pystate[2])
        random.setstate(pystate)

    np = _maybe_numpy()
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])

    torch = _maybe_torch()
    if torch is not None and "torch" in state:
        t = state["torch"]
        if not isinstance(t, torch.Tensor):  # pragma: no cover - defensive
            t = torch.tensor(t, dtype=torch.uint8)
        torch.set_rng_state(t.cpu().to(torch.uint8))
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except Exception:  # pragma: no cover - device count changed
                pass


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - needs workers
    """DataLoader ``worker_init_fn`` giving each worker a distinct, stable seed."""
    torch = _maybe_torch()
    base = 0 if torch is None else int(torch.initial_seed()) % (2**31 - 1)
    seed_all(base + worker_id, deterministic=False)
