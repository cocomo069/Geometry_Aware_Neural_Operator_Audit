"""Non-learned baselines (PLAN.md Phase 3.6, gate G2).

Two predictors, both implementing the CONTEXT.md 7 model interface so that the
*same* :func:`src.eval.harness.evaluate` produces a ``metrics.json`` in the same
frozen schema, with ``model = 'constant'`` / ``'ridge'``:

``ConstantBaseline``
    Train-set mean surface pressure, mean wall shear and mean (CL, CD).  The
    "predict the average" floor every learned model must clear.

``RidgeBaseline``
    Ridge regression from cheap geometric descriptors (NACA parameters parsed
    out of the simulation name) plus the freestream condition to (CL, CD).
    Fields fall back to the train mean, so integrated coefficients and FSC stay
    computable and the two baselines are directly comparable on fields.

Both are fitted in **dataset space** (i.e. on whatever the loader hands over,
normalized or not).  The harness then applies ``dataset.denormalize`` to
predictions *and* ground truth identically, so reported metrics are in physical
units exactly as for learned models.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.eval.harness import batch_get, count_params, evaluate

__all__ = [
    "parse_naca_params",
    "geometry_features",
    "ConstantBaseline",
    "RidgeBaseline",
    "fit_constant",
    "fit_ridge",
    "run_baseline",
    "iter_items",
]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def parse_naca_params(sim_name: str) -> tuple[np.ndarray, int]:
    """Extract NACA shape parameters from an AirfRANS simulation name.

    AirfRANS encodes the case in the name, e.g.
    ``airFoil2D_SST_57.872_6.917_3.16_3.542_11.436``: fields 2 and 3 are the
    inlet velocity and the angle of attack, and the NACA parameters follow from
    field 4 -- **3** of them for a 4-digit airfoil, **4** for a 5-digit one.

    A1's ``src.data.splits.parse_sim_name`` is the doc-verified authority and is
    used whenever it imports; the local fallback below (split on ``_``, keep
    fully-numeric tokens, drop the leading velocity/AoA pair) exists only so
    ``src.eval`` stays importable and testable without ``src.data``.

    Returns
    -------
    (params, n_digits_flag)
        ``params`` is length 4, zero-padded on the right; the flag is 1 for a
        5-digit family (4 shape numbers), else 0.  Unparseable names yield
        zeros and flag 0 rather than raising -- a baseline must never be the
        thing that stops an evaluation.
    """
    try:
        from src.data.splits import parse_sim_name  # noqa: PLC0415

        parsed = parse_sim_name(str(sim_name))
        out = np.zeros(4, dtype=np.float64)
        out[: len(parsed.naca_params)] = parsed.naca_params
        return out, 1 if parsed.n_digits == 5 else 0
    except Exception:
        pass

    nums = [float(t) for t in str(sim_name).split("_") if _NUM_RE.fullmatch(t)]
    shape = nums[2:] if len(nums) > 2 else []
    shape = shape[:4]
    flag = 1 if len(shape) >= 4 else 0
    out = np.zeros(4, dtype=np.float64)
    out[: len(shape)] = shape
    return out, flag


def geometry_features(sim_name: str, cond: Sequence[float]) -> np.ndarray:
    """Feature vector for the ridge baseline: NACA params + freestream.

    ``[p0..p3, five_digit_flag, ux, uy, |u|, aoa, aoa^2, aoa*p0, aoa*p3]``
    where ``aoa = atan2(uy, ux)`` in radians.  The two interactions are the
    first-order aerodynamics: lift is roughly linear in incidence and camber,
    drag roughly quadratic in incidence and linear in thickness.
    """
    params, flag = parse_naca_params(sim_name)
    c = np.asarray(cond, dtype=np.float64).reshape(-1)
    ux = float(c[0]) if c.size > 0 else 0.0
    uy = float(c[1]) if c.size > 1 else 0.0
    umag = float(math.hypot(ux, uy))
    aoa = float(math.atan2(uy, ux)) if umag > 0 else 0.0
    return np.array(
        [*params, float(flag), ux, uy, umag, aoa, aoa * aoa,
         aoa * params[0], aoa * params[3]],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Dataset iteration
# --------------------------------------------------------------------------- #
def iter_items(dataset: Any) -> Iterator[Mapping[str, Any]]:
    """Iterate a map-style dataset (or any iterable) yielding item dicts."""
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for i in range(len(dataset)):
            yield dataset[i]
    else:  # pragma: no cover - iterable datasets
        yield from dataset


def _scalar(item: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    v = batch_get(item, key)
    if v is None:
        return default
    arr = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                     dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else default


def _array(item: Mapping[str, Any], key: str):
    v = batch_get(item, key)
    if v is None:
        return None
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                      dtype=np.float64)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@dataclass
class _BaseBaseline:
    """Common plumbing so baselines can be dropped into the harness verbatim."""

    name: str = "baseline"
    p_mean: float = 0.0
    tau_mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    coef_mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    training: bool = False

    # no-op nn.Module surface -------------------------------------------- #
    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, batch):
        return self.forward(batch)

    @property
    def n_params(self) -> int:
        return 3  # p_mean + 2 tau components

    # ---------------------------------------------------------------- #
    def _n_sims(self, batch) -> int:
        bidx = batch_get(batch, "batch_idx")
        if bidx is not None and bidx.numel():
            return int(bidx.max().item()) + 1
        cond = batch_get(batch, "cond")
        if cond is not None and getattr(cond, "ndim", 0) == 2:
            return int(cond.shape[0])
        return 1

    def _mean_fields(self, batch, n_pts: int, device, dtype):
        import torch

        p = torch.full((n_pts,), float(self.p_mean), device=device, dtype=dtype)
        tau = torch.zeros((n_pts, 2), device=device, dtype=dtype)
        tau[:, 0] = float(self.tau_mean[0])
        tau[:, 1] = float(self.tau_mean[1])
        return p, tau

    def forward(self, batch) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class ConstantBaseline(_BaseBaseline):
    """Predict the training-set mean field and mean coefficients everywhere."""

    name: str = "constant"

    def forward(self, batch) -> dict[str, Any]:
        """CONTEXT.md 7 interface: ``{'p','tau','coef_head'}``."""
        import torch

        pos = batch_get(batch, "surf_pos")
        if pos is None:
            raise KeyError("ConstantBaseline.forward: batch has no 'surf_pos'")
        n_pts = int(pos.shape[0])
        n = self._n_sims(batch)
        p, tau = self._mean_fields(batch, n_pts, pos.device, pos.dtype)
        coef = torch.tensor(self.coef_mean, device=pos.device, dtype=pos.dtype)
        return {"p": p, "tau": tau, "coef_head": coef.reshape(1, 2).expand(n, 2).contiguous()}


@dataclass
class RidgeBaseline(_BaseBaseline):
    """Ridge regression from NACA parameters + freestream to (CL, CD)."""

    name: str = "ridge"
    model: Any = None  # fitted sklearn pipeline
    alpha: float = 1.0
    n_features: int = 11

    @property
    def n_params(self) -> int:
        if self.model is None:
            return 3
        try:
            ridge = self.model[-1]
            return int(np.asarray(ridge.coef_).size + np.asarray(ridge.intercept_).size) + 3
        except Exception:  # pragma: no cover
            return 3

    def predict_coef(self, sim_names: Sequence[str], conds: np.ndarray) -> np.ndarray:
        """``(B, 2)`` array of (CL, CD) predictions."""
        feats = np.stack(
            [geometry_features(sim_names[i], conds[i]) for i in range(len(sim_names))]
        )
        if self.model is None:
            return np.tile(self.coef_mean.reshape(1, 2), (len(sim_names), 1))
        return np.asarray(self.model.predict(feats), dtype=np.float64).reshape(-1, 2)

    def forward(self, batch) -> dict[str, Any]:
        """CONTEXT.md 7 interface; fields are the train mean, coefs from ridge."""
        import torch

        pos = batch_get(batch, "surf_pos")
        if pos is None:
            raise KeyError("RidgeBaseline.forward: batch has no 'surf_pos'")
        n_pts = int(pos.shape[0])
        n = self._n_sims(batch)
        p, tau = self._mean_fields(batch, n_pts, pos.device, pos.dtype)

        cond = batch_get(batch, "cond")
        conds = (
            np.asarray(cond.detach().cpu().numpy(), dtype=np.float64).reshape(n, -1)
            if cond is not None
            else np.zeros((n, 2))
        )
        names = batch_get(batch, "sim_name")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, (list, tuple)) or len(names) != n:
            names = [""] * n
        coef = self.predict_coef(list(names), conds)
        return {
            "p": p,
            "tau": tau,
            "coef_head": torch.tensor(coef, device=pos.device, dtype=pos.dtype),
        }


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _fit_means(train_dataset: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """Point-weighted mean p / tau and sim-weighted mean (CL, CD) on train."""
    p_sum = 0.0
    p_cnt = 0
    tau_sum = np.zeros(2)
    tau_cnt = 0
    coefs: list[list[float]] = []
    for item in iter_items(train_dataset):
        p = _array(item, "surf_p")
        if p is not None and p.size:
            p_sum += float(p.sum())
            p_cnt += int(p.size)
        tau = _array(item, "surf_tau")
        if tau is not None and tau.size:
            t = tau.reshape(-1, 2)
            tau_sum += t.sum(axis=0)
            tau_cnt += t.shape[0]
        coefs.append([_scalar(item, "cl_true"), _scalar(item, "cd_true")])
    arr = np.asarray(coefs, dtype=np.float64) if coefs else np.zeros((1, 2))
    with np.errstate(invalid="ignore"):
        coef_mean = np.nanmean(arr, axis=0)
    coef_mean = np.where(np.isfinite(coef_mean), coef_mean, 0.0)
    return (
        p_sum / p_cnt if p_cnt else 0.0,
        tau_sum / tau_cnt if tau_cnt else np.zeros(2),
        coef_mean,
    )


def fit_constant(train_dataset: Any) -> ConstantBaseline:
    """Fit the constant predictor on the training split."""
    p_mean, tau_mean, coef_mean = _fit_means(train_dataset)
    return ConstantBaseline(p_mean=p_mean, tau_mean=tau_mean, coef_mean=coef_mean)


def fit_ridge(train_dataset: Any, *, alpha: float = 1.0) -> RidgeBaseline:
    """Fit ridge regression ``features -> (CL, CD)`` on the training split.

    Uses ``sklearn`` (``StandardScaler`` + ``Ridge``); falls back to the mean
    predictor when sklearn is unavailable or the split has fewer than three
    usable simulations.
    """
    p_mean, tau_mean, coef_mean = _fit_means(train_dataset)
    feats: list[np.ndarray] = []
    targets: list[list[float]] = []
    for item in iter_items(train_dataset):
        name = batch_get(item, "sim_name") or ""
        cond = _array(item, "cond")
        cl, cd = _scalar(item, "cl_true"), _scalar(item, "cd_true")
        if not (math.isfinite(cl) and math.isfinite(cd)):
            continue
        feats.append(geometry_features(str(name), cond if cond is not None else [0.0, 0.0]))
        targets.append([cl, cd])

    base = RidgeBaseline(p_mean=p_mean, tau_mean=tau_mean, coef_mean=coef_mean, alpha=alpha)
    if len(feats) < 3:
        return base
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:  # pragma: no cover - sklearn missing
        return base
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(np.stack(feats), np.asarray(targets, dtype=np.float64))
    base.model = pipe
    base.n_features = int(np.stack(feats).shape[1])
    return base


# --------------------------------------------------------------------------- #
# End-to-end runner
# --------------------------------------------------------------------------- #
def run_baseline(
    which: str,
    train_dataset: Any,
    test_dataset: Any,
    split_name: str,
    *,
    seed: int = 0,
    tag: str | None = None,
    device: str = "cpu",
    batch_size: int = 1,
    collate_fn: Any = None,
    save_dir: Any = None,
    **evaluate_kwargs: Any,
) -> dict[str, Any]:
    """Fit ``which`` in {'constant','ridge'} on train, evaluate on test.

    Returns the schema-valid metrics dict (also written to ``save_dir`` when
    given).  ``run_id`` follows the frozen ``{model}_{split}_s{seed}`` rule.
    """
    from src.utils.io import build_run_id

    which = which.lower()
    if which == "constant":
        model: _BaseBaseline = fit_constant(train_dataset)
    elif which == "ridge":
        model = fit_ridge(train_dataset)
    else:
        raise ValueError(f"unknown baseline {which!r}; expected 'constant' or 'ridge'")

    run_id = build_run_id(model.name, split_name, seed, tag)
    return evaluate(
        model,
        test_dataset,
        split_name,
        run_meta={
            "run_id": run_id,
            "model": model.name,
            "seed": seed,
            "tag": tag,
            "params": model.n_params,
            "train_time_s": 0.0,
            "epochs": 0,
        },
        device=device,
        batch_size=batch_size,
        collate_fn=collate_fn,
        save_dir=save_dir,
        **evaluate_kwargs,
    )
