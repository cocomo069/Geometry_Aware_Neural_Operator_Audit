"""Deep-ensemble aggregation (spec section 5.6).

Given K independently-trained members of the same architecture we need

    mean(x) = (1/K) sum_k f_k(x)
    var(x)  = (1/(K-1)) sum_k (f_k(x) - mean(x))^2

both per surface point (fields ``p``, ``tau``) and per simulation
(coefficients ``coef_head``), plus the *propagated* coefficient spread obtained
by pushing each member's surface fields through the (linear, differentiable)
force integration of ``src/physics/force_integration.py`` and only then taking
mean/std over members.  That ordering matters: integrate-then-average is what
the spec defines (``Cbar_D = (1/K) sum_k C_D^int[u_hat_k]``), and it is *not*
the same as integrating the mean field when the integrator is used with
non-linear post-processing.

Everything here is numpy-first and torch-agnostic: inputs may be numpy arrays
or torch tensors, outputs are numpy arrays.  Torch is imported lazily and only
where it is genuinely needed (running actual model modules), so this module and
its tests work on a machine where torch is still installing.

Shapes
------
``members``  : sequence of K arrays that broadcast to a common shape S, or a
               single stacked array of shape ``(K, *S)``.
Returned mean/std always have shape ``S``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "EnsembleStats",
    "as_numpy",
    "stack_members",
    "ensemble_mean_std",
    "EnsemblePredictor",
    "propagate",
    "build_model_from_config",
    "load_ensemble_checkpoints",
]

# Field keys produced by every model (CONTEXT.md section 7).
DEFAULT_KEYS = ("p", "tau", "coef_head")


# --------------------------------------------------------------------------
# array plumbing
# --------------------------------------------------------------------------
def as_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / lists / scalars to a float64-safe numpy array.

    Torch is never imported here; we duck-type on ``detach``/``cpu``/``numpy``
    so the module has no hard torch dependency.
    """
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        x = cpu()
    to_numpy = getattr(x, "numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(x)


def stack_members(members: Any) -> np.ndarray:
    """Stack K member predictions into one ``(K, *S)`` array.

    Accepts a sequence of K arrays (each of identical shape S) or an already
    stacked array whose leading axis is the member axis.
    """
    if isinstance(members, np.ndarray) and members.ndim >= 1 and not members.dtype == object:
        # Ambiguity guard: a bare (K, *S) array is taken as already stacked.
        arr = members
    elif isinstance(members, (list, tuple)):
        mats = [as_numpy(m) for m in members]
        if len(mats) == 0:
            raise ValueError("ensemble has zero members")
        shapes = {m.shape for m in mats}
        if len(shapes) != 1:
            raise ValueError(f"ensemble members have mismatched shapes: {sorted(shapes)}")
        arr = np.stack(mats, axis=0)
    else:
        arr = as_numpy(members)
    if arr.ndim < 1:
        raise ValueError("stacked members must have at least a member axis")
    return arr


def ensemble_mean_std(members: Any, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-element ensemble mean and standard deviation.

    Parameters
    ----------
    members : ``(K, *S)`` array or sequence of K arrays of shape S.
    ddof    : delta degrees of freedom; spec section 5.6 uses ``1/(K-1)`` so the
              default is 1.  For K == 1 the std is defined as zeros (a single
              member carries no spread information) rather than NaN.

    Returns
    -------
    (mean, std) both of shape S.
    """
    arr = stack_members(members).astype(np.float64, copy=False)
    k = arr.shape[0]
    if k == 0:
        raise ValueError("ensemble has zero members")
    mean = arr.mean(axis=0)
    if k - ddof <= 0:
        std = np.zeros_like(mean)
    else:
        std = arr.std(axis=0, ddof=ddof)
    return mean, std


@dataclass
class EnsembleStats:
    """Mean / std / raw members for one output key."""

    mean: np.ndarray
    std: np.ndarray
    members: np.ndarray  # (K, *S)

    @property
    def k(self) -> int:
        return int(self.members.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.mean.shape)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_members(cls, members: Any, ddof: int = 1) -> "EnsembleStats":
        arr = stack_members(members).astype(np.float64, copy=False)
        mean, std = ensemble_mean_std(arr, ddof=ddof)
        return cls(mean=mean, std=std, members=arr)


def propagate(members: Any, fn: Callable[[Any], Any], ddof: int = 1) -> EnsembleStats:
    """Apply ``fn`` to each member independently, then aggregate.

    ``fn`` maps one member's prediction to a derived quantity (an array, or a
    scalar).  This is the generic form of "propagation through the force
    integration": pass the integrator (already bound to the geometry) as ``fn``.
    """
    outs = [as_numpy(fn(m)) for m in (members if not isinstance(members, np.ndarray) else list(members))]
    return EnsembleStats.from_members(outs, ddof=ddof)


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------
class EnsemblePredictor:
    """Wrap K callables (models) or K pre-computed prediction dicts.

    Two construction paths:

    * ``EnsemblePredictor(models=[m1, ..., mK])`` where each ``m`` is callable
      on a batch dict and returns ``{"p": ..., "tau": ..., "coef_head": ...}``
      (CONTEXT.md section 7).  ``torch.no_grad`` is used when torch is present.
    * ``EnsemblePredictor.from_arrays({"cd": (K, n) array, ...})`` for offline
      aggregation of predictions dumped to disk, and for tests.

    ``ddof`` defaults to 1 per spec section 5.6.
    """

    def __init__(
        self,
        models: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None,
        *,
        ddof: int = 1,
        keys: Sequence[str] | None = None,
        member_ids: Sequence[Any] | None = None,
    ) -> None:
        self.models = list(models) if models is not None else []
        self.ddof = int(ddof)
        self.keys = tuple(keys) if keys is not None else None
        self.member_ids = list(member_ids) if member_ids is not None else list(range(len(self.models)))
        self._arrays: dict[str, np.ndarray] | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, Any],
        *,
        ddof: int = 1,
        member_ids: Sequence[Any] | None = None,
    ) -> "EnsemblePredictor":
        """Build from a mapping ``key -> (K, *S)`` stacked member predictions."""
        stacked = {k: stack_members(v).astype(np.float64, copy=False) for k, v in arrays.items()}
        if not stacked:
            raise ValueError("from_arrays needs at least one key")
        ks = {v.shape[0] for v in stacked.values()}
        if len(ks) != 1:
            raise ValueError(f"inconsistent member counts across keys: {ks}")
        obj = cls(models=None, ddof=ddof, keys=tuple(stacked), member_ids=member_ids)
        obj._arrays = stacked
        obj.member_ids = list(member_ids) if member_ids is not None else list(range(ks.pop()))
        return obj

    @property
    def k(self) -> int:
        if self._arrays is not None:
            return int(next(iter(self._arrays.values())).shape[0])
        return len(self.models)

    # -- prediction --------------------------------------------------------
    def member_outputs(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        raw: bool = False,
    ) -> list[dict[str, Any]]:
        """Run every member and return a list of K output dicts.

        ``raw=True`` skips the numpy conversion and hands back whatever the
        models returned (torch tensors, normally).  That matters for
        :meth:`propagate_forces`, which feeds the outputs straight into A2's
        torch-native ``integrate_forces``.
        """
        if self._arrays is not None:
            k = self.k
            return [{key: v[i] for key, v in self._arrays.items()} for i in range(k)]
        if batch is None:
            raise ValueError("a batch is required when the predictor wraps live models")
        if not self.models:
            raise ValueError("ensemble has zero members")
        ctx = _maybe_no_grad()
        outs: list[dict[str, Any]] = []
        with ctx:
            for model in self.models:
                _maybe_eval(model)
                out = model(batch)
                if not isinstance(out, Mapping):
                    raise TypeError(f"model returned {type(out)!r}, expected a mapping")
                keys = self.keys if self.keys is not None else tuple(out.keys())
                outs.append(
                    {key: (out[key] if raw else as_numpy(out[key])) for key in keys if key in out}
                )
        return outs

    def predict(self, batch: Mapping[str, Any] | None = None) -> dict[str, EnsembleStats]:
        """Return ``{key: EnsembleStats}`` with per-element mean and std.

        For ``p`` (shape ``(sum Ns,)``) and ``tau`` (``(sum Ns, 2)``) this is the
        per-point spread; for ``coef_head`` (``(B, 2)``) it is the per-simulation
        coefficient spread.
        """
        outs = self.member_outputs(batch)
        keys = self.keys if self.keys is not None else tuple(outs[0].keys())
        stats: dict[str, EnsembleStats] = {}
        for key in keys:
            if any(key not in o for o in outs):
                continue
            stats[key] = EnsembleStats.from_members([o[key] for o in outs], ddof=self.ddof)
        return stats

    def predict_mean_std(self, batch: Mapping[str, Any] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Flattened view of :meth:`predict`: ``{key: {"mean":..., "std":...}}``."""
        return {k: v.as_dict() for k, v in self.predict(batch).items()}

    # -- force propagation -------------------------------------------------
    def propagate_forces(
        self,
        batch: Mapping[str, Any] | None,
        integrate_fn: Callable[..., Mapping[str, Any]],
        *,
        p_key: str = "p",
        tau_key: str = "tau",
        denormalize: Callable[[str, Any], Any] | None = None,
        **integrate_kwargs: Any,
    ) -> dict[str, EnsembleStats]:
        """Integrate each member's surface fields, then aggregate over members.

        ``integrate_fn`` must have the frozen signature of
        ``src.physics.force_integration.integrate_forces``::

            integrate_fn(p, tau, normal, ds, cond, batch_idx=None, ...) -> dict

        We do not import it here (it belongs to A2 and may not exist yet) --
        callers pass it in.  Any extra kwargs (``rho``, ``a_ref``) are forwarded.

        ``denormalize`` (optional) maps ``(field_name, tensor) -> tensor`` and is
        applied to ``p``/``tau`` before integration so coefficients come out in
        physical units (CONTEXT.md section 9: all evaluation is denormalized).

        Returns ``{"cl": EnsembleStats, "cd": EnsembleStats, ...}`` -- one entry
        per key the integrator returns, each of shape ``(B,)``.
        """
        if batch is None:
            raise ValueError("force propagation needs the geometry from the batch")
        # raw=True: hand the models' own tensors to the (torch-native,
        # differentiable) integrator instead of numpy copies.
        outs = self.member_outputs(batch, raw=True)
        geom = {k: batch[k] for k in ("surf_normal", "surf_ds", "cond") if k in batch}
        missing = {"surf_normal", "surf_ds", "cond"} - set(geom)
        if missing:
            raise KeyError(f"batch is missing geometry keys required for integration: {sorted(missing)}")
        batch_idx = batch.get("batch_idx", None)

        per_member: dict[str, list[np.ndarray]] = {}
        for out in outs:
            p = out[p_key]
            tau = out[tau_key]
            if denormalize is not None:
                p = denormalize("p", p)
                tau = denormalize("tau", tau)
            res = integrate_fn(
                p,
                tau,
                geom["surf_normal"],
                geom["surf_ds"],
                geom["cond"],
                batch_idx=batch_idx,
                **integrate_kwargs,
            )
            if not isinstance(res, Mapping):
                raise TypeError("integrate_fn must return a mapping like {'cl':..., 'cd':...}")
            for key, val in res.items():
                per_member.setdefault(key, []).append(np.atleast_1d(as_numpy(val)))
        return {key: EnsembleStats.from_members(vals, ddof=self.ddof) for key, vals in per_member.items()}

    def propagate(self, fn: Callable[[Mapping[str, np.ndarray]], Any], batch=None) -> EnsembleStats:
        """Apply an arbitrary per-member callable to the member output dicts."""
        outs = self.member_outputs(batch)
        return EnsembleStats.from_members([as_numpy(fn(o)) for o in outs], ddof=self.ddof)


def _maybe_no_grad():
    try:  # late import: torch may still be installing
        import torch
    except Exception:  # pragma: no cover - exercised only without torch
        from contextlib import nullcontext

        return nullcontext()
    return torch.no_grad()


def _maybe_eval(model: Any) -> None:
    ev = getattr(model, "eval", None)
    if callable(ev):
        try:
            ev()
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------
#: Model-name -> import path candidates.  Kept as a table so that adding a
#: fourth architecture is a one-line change and so that the lookup does not
#: import ``src.models`` at module import time (A3 owns that package).
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gnn": ("src.models.gnn", "GNNSurrogate"),
    "sdf_fno": ("src.models.sdf_fno", "SDFFNOSurrogate"),
    "transolver": ("src.models.geo_transformer", "GeoTransformerSurrogate"),
    # aliases mirroring src/models/__init__.py::_ALIASES
    "m1": ("src.models.gnn", "GNNSurrogate"),
    "m2": ("src.models.sdf_fno", "SDFFNOSurrogate"),
    "m3": ("src.models.geo_transformer", "GeoTransformerSurrogate"),
    "geo_transformer": ("src.models.geo_transformer", "GeoTransformerSurrogate"),
}


def build_model_from_config(config: Mapping[str, Any]) -> Any:
    """Instantiate a model from a run config dict (late import of ``src.models``).

    Resolution order:

    1. ``src.models.build_model`` if A3 exposes a factory (preferred).  A3's
       signature is ``build_model(name, config)``; we also accept a
       single-argument ``build_model(config)`` so the call site survives a
       future signature change.
    2. ``MODEL_REGISTRY`` lookup on ``config["model"]["name"]`` (or
       ``config["model"]``/``config["name"]`` if the config is flatter).

    Raises a descriptive ``ImportError``/``KeyError`` rather than guessing.
    """
    try:
        import importlib

        models_pkg = importlib.import_module("src.models")
    except Exception as exc:  # pragma: no cover - depends on A3 landing
        raise ImportError(
            "src.models is not importable yet; pass explicit model objects to "
            f"EnsemblePredictor instead ({exc})"
        ) from exc

    factory = getattr(models_pkg, "build_model", None)
    if callable(factory):
        import inspect

        name = _config_model_name(config)
        model_cfg = config.get("model") if isinstance(config, Mapping) else None
        if not isinstance(model_cfg, Mapping):
            model_cfg = {}
        model_cfg = {k: v for k, v in model_cfg.items() if k != "name"}
        try:
            n_positional = len(
                [
                    p
                    for p in inspect.signature(factory).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            )
        except (TypeError, ValueError):  # pragma: no cover - builtins
            n_positional = 2
        if n_positional >= 2:
            return factory(name, model_cfg)
        return factory(config)

    name = _config_model_name(config)
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model name {name!r}; known: {sorted(MODEL_REGISTRY)} "
            "(or expose src.models.build_model)"
        )
    import importlib

    mod_path, cls_name = MODEL_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    model_cfg = config.get("model", config) if isinstance(config, Mapping) else config
    return cls(model_cfg)


def _config_model_name(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    model = config.get("model")
    if isinstance(model, Mapping):
        name = model.get("name")
        if name:
            return str(name)
    if isinstance(model, str):
        return model
    name = config.get("name")
    if name:
        return str(name)
    raise KeyError("config has no model name (looked at config['model']['name'], config['model'], config['name'])")


def load_ensemble_checkpoints(
    checkpoints: Iterable[str | Path],
    config: Mapping[str, Any],
    *,
    map_location: str = "cpu",
    state_key: str = "model",
    strict: bool = True,
    ddof: int = 1,
) -> EnsemblePredictor:
    """Load K checkpoints of the *same* architecture into an EnsemblePredictor.

    ``checkpoints`` are paths to ``checkpoints/<run_id>/best.pt`` files written
    by ``scripts/train.py`` (CONTEXT.md section 10: the checkpoint dict carries
    model+optim+epoch+RNG, so we pull ``state_key`` out of it; a bare state_dict
    is also accepted).

    All members share ``config``: the ensemble is over seeds/initializations,
    not architectures (spec section 5.6).
    """
    import importlib

    torch = importlib.import_module("torch")  # late import by design

    paths = [Path(p) for p in checkpoints]
    if not paths:
        raise ValueError("no checkpoints given")
    models = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        state = blob.get(state_key, blob) if isinstance(blob, Mapping) else blob
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        model = build_model_from_config(config)
        model.load_state_dict(state, strict=strict)
        model.eval()
        models.append(model)
    return EnsemblePredictor(models, ddof=ddof, member_ids=[p.parent.name for p in paths])
