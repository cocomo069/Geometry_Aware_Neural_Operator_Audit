"""Evaluation harness -- the *only* producer of ``metrics.json`` (CONTEXT.md 9).

``evaluate(model, dataset, split_name, ...)`` runs a model over a split and
returns a dict matching the frozen schema exactly, validated by
:mod:`src.eval.schema` before it is returned.

Contracts this harness relies on
--------------------------------
* model (CONTEXT.md 7): ``forward(batch: dict) -> {"p": (Ns,), "tau": (Ns,2),
  "coef_head": (B,2)}`` in **normalized** space, ``coef_head = (CL, CD)``.
* batch (CONTEXT.md 4): concatenated points with ``batch_idx`` (long, (Ns,)),
  keys ``surf_pos, surf_normal, surf_ds, surf_p, surf_tau, cond, cl_true,
  cd_true, sim_name``.
* physics (CONTEXT.md 6): ``integrate_forces(p, tau, normal, ds, cond,
  batch_idx=None, rho=1.0, a_ref=1.0) -> dict(cl, cd, fx, fy)``.
* loader: ``dataset.denormalize(field_name, tensor)`` maps normalized ->
  physical units.  **All metrics are computed in physical units.**

Cross-module imports are *late* and guarded, so this module imports (and its
tests run) before ``src/data``, ``src/models`` or ``src/physics`` exist.
"""

from __future__ import annotations

import importlib
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.eval.metrics import MetricAccumulator, as_numpy
from src.eval.schema import assert_valid_metrics, empty_metrics

__all__ = [
    "ModuleNotReadyError",
    "evaluate",
    "count_params",
    "identity_denormalize",
    "reflect_batch",
    "batch_get",
    "batch_keys",
]

SURF_KEYS = ("surf_pos", "surf_normal", "surf_ds", "surf_p", "surf_tau")


class ModuleNotReadyError(RuntimeError):
    """A sibling module (data/models/physics) is not importable yet."""


# --------------------------------------------------------------------------- #
# Batch access helpers -- tolerate dict batches and dataclass-like Batch objects
# --------------------------------------------------------------------------- #
def batch_get(batch: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from a mapping-like or attribute-like batch."""
    if isinstance(batch, Mapping):
        return batch.get(key, default)
    return getattr(batch, key, default)


def batch_keys(batch: Any) -> list[str]:
    """List the batch's field names (mapping keys or public attributes)."""
    if isinstance(batch, Mapping):
        return list(batch.keys())
    if hasattr(batch, "__dataclass_fields__"):
        return list(batch.__dataclass_fields__)
    return [k for k in vars(batch) if not k.startswith("_")]


def _batch_set(batch: Any, key: str, value: Any) -> None:
    if isinstance(batch, dict):
        batch[key] = value
    else:
        setattr(batch, key, value)


def _shallow_copy(batch: Any) -> Any:
    import copy as _copy

    return dict(batch) if isinstance(batch, Mapping) else _copy.copy(batch)


def batch_to(batch: Any, device: Any) -> Any:
    """Move every tensor in the batch to ``device`` (non-tensors untouched)."""
    import torch

    def move(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            return type(v)(x.to(device) for x in v)
        return v

    if isinstance(batch, Mapping):
        return {k: move(v) for k, v in batch.items()}
    out = _shallow_copy(batch)
    for k in batch_keys(batch):
        _batch_set(out, k, move(batch_get(batch, k)))
    return out


# --------------------------------------------------------------------------- #
# Denormalization
# --------------------------------------------------------------------------- #
def identity_denormalize(name: str, tensor: Any) -> Any:
    """Denormalizer for models that already predict physical units (baselines)."""
    return tensor


class _Denormalizer:
    """Wrap ``dataset.denormalize`` with per-field fallback to the identity.

    Fields the loader does not know about (typically ``cl``/``cd``, which the
    cache stores raw) pass through unchanged, and the fallback is remembered so
    we do not pay the exception cost per batch.
    """

    def __init__(self, fn: Callable[[str, Any], Any] | None):
        self.fn = fn
        self._passthrough: set[str] = set()

    def __call__(self, name: str, tensor: Any) -> Any:
        if self.fn is None or name in self._passthrough or tensor is None:
            return tensor
        try:
            out = self.fn(name, tensor)
        except Exception:
            self._passthrough.add(name)
            return tensor
        return tensor if out is None else out


def _resolve_denormalize(dataset: Any, denormalize: Any) -> _Denormalizer:
    if denormalize is None:
        fn = getattr(dataset, "denormalize", None)
        return _Denormalizer(fn if callable(fn) else None)
    if denormalize is False:
        return _Denormalizer(None)
    if callable(denormalize):
        return _Denormalizer(denormalize)
    raise TypeError(f"denormalize must be None, False or callable, got {denormalize!r}")


# --------------------------------------------------------------------------- #
# Late imports
# --------------------------------------------------------------------------- #
def _resolve_integrate(integrate_fn: Any) -> Callable[..., Mapping[str, Any]] | None:
    """Resolve the force-integration callable (CONTEXT.md 6)."""
    if callable(integrate_fn):
        return integrate_fn
    if integrate_fn is False:
        return None
    try:
        mod = importlib.import_module("src.physics.force_integration")
    except Exception as exc:
        warnings.warn(
            "src.physics.force_integration is not importable "
            f"({type(exc).__name__}: {exc}); integrated coefficients, FSC and "
            "cl_rel/cd_rel will be null in metrics.json.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    fn = getattr(mod, "integrate_forces", None)
    if not callable(fn):  # pragma: no cover - defensive
        warnings.warn(
            "src.physics.force_integration has no integrate_forces(); "
            "integrated coefficients will be null.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return fn


def _resolve_collate(dataset: Any, collate_fn: Any) -> Callable | None:
    if callable(collate_fn):
        return collate_fn
    fn = getattr(dataset, "collate", None)
    if callable(fn):
        return fn
    try:
        mod = importlib.import_module("src.data.airfrans_loader")
    except Exception:
        return None
    fn = getattr(mod, "collate", None)
    return fn if callable(fn) else None


def count_params(model: Any, trainable_only: bool = False) -> int | None:
    """Number of parameters, or ``None`` for objects without ``parameters()``."""
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return int(sum(p.numel() for p in params() if (p.requires_grad or not trainable_only)))
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# Symmetry / antisymmetry (spec 5.5)
# --------------------------------------------------------------------------- #
def reflect_batch(batch: Any) -> Any:
    """Reflect a batch about the x-axis: ``y -> -y`` on positions and vectors.

    Reflecting the geometry *and* the freestream condition maps the problem to
    its mirror image, so this is a valid equivariance probe for **any** case,
    not only symmetric airfoils.  Point ordering is preserved, so residuals can
    be taken index-wise.  Returns a shallow copy; the input is untouched.
    """
    import torch

    out = _shallow_copy(batch)
    for key in ("surf_pos", "surf_normal", "surf_tau", "surf_tau_phys",
                "vol_pos", "vol_u", "cond", "cond_phys"):
        v = batch_get(batch, key)
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[-1] == 2:
            w = v.clone()
            w[..., 1] = -w[..., 1]
            _batch_set(out, key, w)
    return out


def reflect_prediction(pred: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ``R#`` to a prediction: p invariant, tau_y negated, CL negated.

    Uses ``src.geometry.symmetry.reflect_prediction`` when importable so the
    convention has a single owner (A2); the local copy keeps this module
    self-contained.
    """
    import torch

    try:
        mod = importlib.import_module("src.geometry.symmetry")
        fn = getattr(mod, "reflect_prediction", None)
        if callable(fn):
            return fn(dict(pred))
    except Exception:
        pass
    out = dict(pred)
    tau = out.get("tau")
    if isinstance(tau, torch.Tensor):
        t = tau.clone()
        t[..., 1] = -t[..., 1]
        out["tau"] = t
    coef = out.get("coef_head")
    if isinstance(coef, torch.Tensor):
        c = coef.clone()
        c[..., 0] = -c[..., 0]
        out["coef_head"] = c
    return out


def _symmetry_residual(pred, pred_r, batch_idx, n_graphs: int, denorm):
    """Per-sim ``||u - R# u(Rg,Rc)||_2 / ||u||_2`` in **physical** units.

    ``pred_r`` must already have ``R#`` applied.  Each surface point
    contributes the 3-vector ``(p, tau_x, tau_y)``; denormalizing first matters
    because the pressure standardisation has a non-zero mean, so the ratio is
    not the same in normalized space (D-014).
    """
    import torch

    p, pr = pred.get("p"), pred_r.get("p")
    if p is None or pr is None:
        return None
    cols = [denorm("surf_p", p).reshape(-1, 1)]
    cols_r = [denorm("surf_p", pr).reshape(-1, 1)]
    tau, tau_r = pred.get("tau"), pred_r.get("tau")
    if tau is not None and tau_r is not None:
        cols.append(denorm("surf_tau", tau))
        cols_r.append(denorm("surf_tau", tau_r))
    u = torch.cat(cols, dim=1).double()
    u_r = torch.cat(cols_r, dim=1).double()
    diff_sq = (u - u_r).pow(2).sum(dim=1)
    ref_sq = u.pow(2).sum(dim=1)
    if batch_idx is None:
        den = float(ref_sq.sum().sqrt())
        return None if den == 0.0 else [float(diff_sq.sum().sqrt()) / den]
    num = torch.zeros(n_graphs, dtype=diff_sq.dtype, device=diff_sq.device)
    den = torch.zeros_like(num)
    num.index_add_(0, batch_idx, diff_sq)
    den.index_add_(0, batch_idx, ref_sq)
    return as_numpy(num.sqrt() / (den.sqrt() + 1e-12))


#: Geometry-derived caches a batch may carry (D-017).  They are dropped from
#: the *reflected* batch so the model recomputes them from the mirrored
#: geometry instead of silently reusing values built for the original one.
CACHED_DERIVED_KEYS = ("grid_sdf", "grid_mask", "grid_feat",
                       "edge_index", "edge_attr", "knn_idx")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def evaluate(
    model: Any,
    dataset: Any,
    split_name: str,
    *,
    run_meta: Mapping[str, Any] | None = None,
    device: Any = "cpu",
    batch_size: int = 1,
    collate_fn: Callable | None = None,
    denormalize: Any = None,
    integrate_fn: Any = "auto",
    max_batches: int | None = None,
    compute_symmetry: bool = True,
    symmetry_max_sims: int | None = 64,
    spearman_source: str = "int",
    rho: float = 1.0,
    a_ref: float = 1.0,
    save_dir: str | Path | None = None,
    validate: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate ``model`` on ``dataset`` and return a schema-valid metrics dict.

    Parameters
    ----------
    model:
        Anything implementing CONTEXT.md 7 ``forward(batch) -> dict``.
    dataset:
        Torch dataset (or any sequence of item dicts) for ``split_name``'s test
        portion.  If it exposes ``denormalize``, it is used automatically.
    split_name:
        Name written into ``metrics.json['split']`` (``full``, ``aoa``, ...).
    run_meta:
        ``run_id, model, seed, tag, params, train_time_s, epochs`` overrides.
    device:
        ``'cpu'``, ``'cuda'`` or a ``torch.device``.  Peak CUDA memory and
        synchronised timings are recorded automatically on CUDA.
    integrate_fn:
        ``'auto'`` (import ``src.physics.force_integration``), a callable with
        the CONTEXT.md 6 signature, or ``False`` to skip integration.
    denormalize:
        ``None`` -> use ``dataset.denormalize``; a callable ``(name, tensor)``;
        or ``False`` for models already in physical units (baselines).
    save_dir:
        If given, writes ``metrics.json`` and ``per_sim.csv`` there.

    Returns
    -------
    dict
        ``metrics.json`` content (CONTEXT.md 9), validated unless
        ``validate=False``.
    """
    import torch

    dev = torch.device(device)
    denorm = _resolve_denormalize(dataset, denormalize)
    integrate = _resolve_integrate(integrate_fn)
    collate = _resolve_collate(dataset, collate_fn)

    model_was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        try:
            model.to(dev)
        except Exception:  # pragma: no cover - non-module baselines
            pass

    loader = _make_loader(dataset, batch_size=batch_size, collate=collate)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

    acc = MetricAccumulator()
    n_sims = 0
    forward_s = 0.0
    sym_done = 0
    warmed_up = False

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = batch_to(batch, dev)

            if not warmed_up:  # exclude lazy-init / cudnn autotune from timing
                try:
                    model(batch)
                except Exception as exc:
                    raise ModuleNotReadyError(
                        f"model forward failed on batch 0: {type(exc).__name__}: {exc}"
                    ) from exc
                warmed_up = True
                if dev.type == "cuda":
                    torch.cuda.synchronize()

            t0 = time.perf_counter()
            pred = model(batch)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            forward_s += time.perf_counter() - t0

            if not isinstance(pred, Mapping):
                raise TypeError(
                    "model.forward must return a dict with keys p/tau/coef_head "
                    f"(CONTEXT.md 7), got {type(pred).__name__}"
                )

            sym_res_b, antisym_b = _maybe_symmetry(
                model, batch, pred, compute_symmetry, symmetry_max_sims, sym_done, denorm
            )
            if sym_res_b is not None:
                sym_done += _n_sims_in(batch, pred)

            n_sims += _accumulate_batch(
                acc, batch, pred, denorm, integrate, rho, a_ref, sym_res_b, antisym_b
            )
            if progress and bi % 25 == 0:
                print(f"  [eval] batch {bi}  sims={n_sims}", flush=True)

    peak_mb = None
    if dev.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 2)

    if model_was_training and hasattr(model, "train"):
        model.train()

    meta = dict(run_meta or {})
    metrics = empty_metrics(
        run_id=meta.get("run_id", ""),
        model=str(meta.get("model", getattr(model, "name", type(model).__name__))),
        split=str(split_name),
        seed=int(meta.get("seed", 0)),
        tag=meta.get("tag"),
        params=meta.get("params", count_params(model)),
        train_time_s=meta.get("train_time_s"),
        epochs=meta.get("epochs"),
        device=str(dev),
    )
    metrics["field"] = acc.field_block()
    metrics["coef"] = acc.coef_block(spearman_source=spearman_source)
    metrics["consistency"] = acc.consistency_block()
    metrics["cost"] = {
        "infer_ms_per_sim": (1000.0 * forward_s / n_sims) if n_sims else None,
        "peak_mem_mb": peak_mb,
    }
    metrics["n_sims"] = n_sims
    for extra in ("dataset", "notes"):
        if extra in meta:
            metrics[extra] = meta[extra]

    metrics = _nan_to_none(metrics)
    if validate:
        assert_valid_metrics(metrics)

    if save_dir is not None:
        from src.utils.io import ensure_dir, write_csv, write_json

        out = ensure_dir(save_dir)
        write_json(out / "metrics.json", metrics)
        rows = acc.per_sim_rows()
        if rows:
            write_csv(out / "per_sim.csv", rows)
    return metrics


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _make_loader(dataset: Any, *, batch_size: int, collate: Callable | None):
    """Build a deterministic, single-process DataLoader (num_workers=0)."""
    import torch
    from torch.utils.data import DataLoader

    if isinstance(dataset, DataLoader):
        return dataset
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": 0,
    }
    if collate is not None:
        kwargs["collate_fn"] = collate
    try:
        return DataLoader(dataset, **kwargs)
    except Exception:  # pragma: no cover - exotic dataset types
        return [dataset[i] for i in range(len(dataset))]


def _n_sims_in(batch: Any, pred: Mapping[str, Any]) -> int:
    coef = pred.get("coef_head")
    if coef is not None and getattr(coef, "ndim", 0) == 2:
        return int(coef.shape[0])
    bidx = batch_get(batch, "batch_idx")
    if bidx is not None and getattr(bidx, "numel", lambda: 0)():
        return int(bidx.max().item()) + 1
    cond = batch_get(batch, "cond")
    if cond is not None and getattr(cond, "ndim", 0) == 2:
        return int(cond.shape[0])
    return 1


def _sim_names(batch: Any, n: int) -> list[str]:
    names = batch_get(batch, "sim_name")
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)) and len(names) == n:
        return [str(x) for x in names]
    return [""] * n


def _maybe_symmetry(model, batch, pred, enabled, max_sims, done, denorm):
    """Return ``(sym_residual, per-sim antisym gap array)`` or ``(None, None)``."""
    if not enabled:
        return None, None
    if max_sims is not None and done >= max_sims:
        return None, None
    try:
        raw = model(_reflect(batch))
        if not isinstance(raw, Mapping):  # pragma: no cover
            return None, None
        pred_r = reflect_prediction(raw)  # R# applied: compare like with like
        n = _n_sims_in(batch, pred)
        sym = _symmetry_residual(pred, pred_r, batch_get(batch, "batch_idx"), n, denorm)

        coef = pred.get("coef_head")
        coef_r = pred_r.get("coef_head")
        antisym = None
        if coef is not None and coef_r is not None:
            # R# already negated CL_r, so equivariance means CL_r == CL, i.e.
            # the raw heads satisfy CL(-alpha) = -CL(alpha) (spec 5.5).
            cl = as_numpy(denorm("cl_true", coef.reshape(n, -1)[:, 0]))
            cl_r = as_numpy(denorm("cl_true", coef_r.reshape(n, -1)[:, 0]))
            antisym = abs(cl - cl_r)
        return sym, antisym
    except Exception as exc:
        warnings.warn(
            f"symmetry check skipped ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None


def _reflect(batch: Any) -> Any:
    """Reflect via ``src.geometry.symmetry`` when available, else locally."""
    try:
        mod = importlib.import_module("src.geometry.symmetry")
        fn = getattr(mod, "reflect_batch", None)
        if callable(fn):
            return fn(batch)
    except Exception:
        pass
    return reflect_batch(batch)


def _accumulate_batch(acc, batch, pred, denorm, integrate, rho, a_ref, sym_res, antisym):
    """Denormalize one batch, integrate forces, split per sim, record metrics."""
    import torch

    n = _n_sims_in(batch, pred)
    names = _sim_names(batch, n)

    p_pred = pred.get("p")
    tau_pred = pred.get("tau")
    coef_head = pred.get("coef_head")

    p_pred_d = denorm("surf_p", p_pred) if p_pred is not None else None
    tau_pred_d = denorm("surf_tau", tau_pred) if tau_pred is not None else None

    p_true_d = _phys(batch, "surf_p", denorm)
    tau_true_d = _phys(batch, "surf_tau", denorm)

    cl_head = cd_head = None
    if coef_head is not None:
        coef = coef_head.reshape(n, -1)
        cl_head = as_numpy(denorm("cl_true", coef[:, 0]))
        cd_head = as_numpy(denorm("cd_true", coef[:, 1]))

    normal = batch_get(batch, "surf_normal")
    ds = batch_get(batch, "surf_ds")
    cond = _phys(batch, "cond", denorm)  # integration needs physical freestream
    bidx = batch_get(batch, "batch_idx")
    if bidx is None and p_pred_d is not None:
        bidx = torch.zeros(p_pred_d.shape[0], dtype=torch.long, device=p_pred_d.device)

    cl_int = cd_int = None
    if (
        integrate is not None
        and p_pred_d is not None
        and tau_pred_d is not None
        and normal is not None
        and ds is not None
        and cond is not None
    ):
        try:
            forces = integrate(
                p_pred_d, tau_pred_d, normal, ds, cond,
                batch_idx=bidx, rho=rho, a_ref=a_ref,
            )
            cl_int = as_numpy(forces["cl"]).reshape(-1)
            cd_int = as_numpy(forces["cd"]).reshape(-1)
        except Exception as exc:
            warnings.warn(
                f"integrate_forces failed ({type(exc).__name__}: {exc}); "
                "integrated coefficients null for this batch.",
                RuntimeWarning,
                stacklevel=2,
            )

    cl_true = _as_vec(_phys(batch, "cl_true", denorm), n)
    cd_true = _as_vec(_phys(batch, "cd_true", denorm), n)

    for i in range(n):
        if bidx is not None and p_pred_d is not None:
            mask = bidx == i
            slc = lambda t: (t[mask] if t is not None else None)  # noqa: E731
        else:
            slc = lambda t: t  # noqa: E731
        acc.add(
            sim_name=names[i],
            p_pred=slc(p_pred_d),
            p_true=slc(p_true_d),
            tau_pred=slc(tau_pred_d),
            tau_true=slc(tau_true_d),
            cl_head=_at(cl_head, i),
            cd_head=_at(cd_head, i),
            cl_int=_at(cl_int, i),
            cd_int=_at(cd_int, i),
            cl_true=_at(cl_true, i),
            cd_true=_at(cd_true, i),
            sym_residual=_at(sym_res, i),
            antisym_cl_gap=_at(antisym, i),
        )
    return n


def _phys(batch: Any, key: str, denorm) -> Any:
    """Physical-unit ground truth for ``key``.

    A1's loader stashes pre-normalisation copies as ``<key>_phys`` (surf_p,
    surf_tau, cond, cl_true, cd_true), which is exact; otherwise we invert the
    normalisation.  ``None`` when the batch carries neither.
    """
    val = batch_get(batch, f"{key}_phys")
    if val is not None:
        return val
    val = batch_get(batch, key)
    return None if val is None else denorm(key, val)


def _as_vec(value: Any, n: int):
    if value is None:
        return None
    arr = as_numpy(value).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        import numpy as np

        return np.repeat(arr, n)
    return arr


def _at(arr: Any, i: int) -> float:
    if arr is None:
        return math.nan
    try:
        return float(arr[i])
    except (IndexError, TypeError):
        return math.nan


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
