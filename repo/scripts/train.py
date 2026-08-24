"""Training entry point (CONTEXT.md 10, FROZEN).

    .venv/Scripts/python.exe -m scripts.train --config configs/gnn.yaml split=full seed=0

Behaviour required by CONTEXT.md 10 and implemented here:

* plain YAML config + dotted CLI overrides (``src/utils/config.py``);
* fp32 only, Adam + cosine schedule, gradient clip 1.0 (no AMP: Pascal);
* a checkpoint **every epoch** to ``checkpoints/<run_id>/last.pt`` carrying
  model + optimizer + scheduler + epoch + RNG states, with **auto-resume**;
* ``best.pt`` on the validation metric (validation = the split's ``cal`` list,
  which never overlaps train or test, CONTEXT.md 5);
* ``results/<run_id>/history.csv`` per epoch, no wandb;
* ends by invoking the eval harness -> ``results/<run_id>/metrics.json``.

Determinism of resume: the training DataLoader is rebuilt each epoch with a
generator seeded ``f(seed, epoch)``, so batch order depends only on the epoch
number, never on how many times the run was interrupted; everything else is
covered by the saved RNG states.  ``tests/test_infra.py`` asserts that 2 epochs
straight and 1 + resume + 1 give bit-identical parameters.

Cross-module imports (``src.data``, ``src.models``, ``src.physics``) are late
and raise a clear "module not ready" error, so this file is importable while
sibling agents are still writing theirs.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow `python scripts/train.py` from anywhere
    sys.path.insert(0, str(_ROOT))

from src.utils.config import get_in, load_config, save_config  # noqa: E402
from src.utils.io import (  # noqa: E402
    append_csv_row,
    build_run_id,
    checkpoint_dir,
    results_dir,
)
from src.utils.seed import get_rng_state, seed_all, set_rng_state  # noqa: E402

__all__ = [
    "FALLBACK_REGISTRY",
    "HISTORY_FIELDS",
    "ModuleNotReadyError",
    "build_model",
    "build_datasets",
    "build_optimizer",
    "build_scheduler",
    "fallback_total_loss",
    "resolve_loss_fn",
    "make_loader",
    "save_checkpoint",
    "load_checkpoint",
    "train_model",
    "resolve_device",
    "parse_cli",
    "main",
]

EPS = 1e-8


class ModuleNotReadyError(RuntimeError):
    """A sibling module (src.data / src.models) is missing or unusable."""


#: ``model.name`` -> ``module:ClassName``.  Used when ``src/models/__init__.py``
#: exposes neither ``MODEL_REGISTRY`` nor ``build_model`` (A3 owns that file).
FALLBACK_REGISTRY: dict[str, str] = {
    "gnn": "src.models.gnn:GNNSurrogate",
    "m1": "src.models.gnn:GNNSurrogate",
    "sdf_fno": "src.models.sdf_fno:SDFFNOSurrogate",
    "gino": "src.models.sdf_fno:SDFFNOSurrogate",
    "m2": "src.models.sdf_fno:SDFFNOSurrogate",
    "transolver": "src.models.geo_transformer:GeoTransformerSurrogate",
    "geo_transformer": "src.models.geo_transformer:GeoTransformerSurrogate",
    "m3": "src.models.geo_transformer:GeoTransformerSurrogate",
}

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_p_rel_l2",
    "train_tau_rel_l2",
    "train_head",
    "train_force",
    "train_sym",
    "val_loss",
    "val_p_rel_l2",
    "val_tau_rel_l2",
    "val_cl_mae",
    "val_cd_mae",
    "lr",
    "epoch_time_s",
)


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(spec: str | None = "auto"):
    """``'auto'`` -> cuda when available, else cpu.  Returns a ``torch.device``."""
    import torch

    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _import_target(target: str):
    mod_name, _, cls_name = str(target).partition(":")
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, cls_name, None)
    if obj is None:
        raise ModuleNotReadyError(f"{mod_name} has no attribute {cls_name!r}")
    return obj


def build_model(cfg: Mapping[str, Any]):
    """Instantiate ``cfg['model']['name']`` with the rest of the model section.

    Resolution order:
      1. ``src.models.build_model(name, model_cfg)`` if defined;
      2. ``src.models.MODEL_REGISTRY[name]`` (class or ``'mod:Cls'`` string);
      3. :data:`FALLBACK_REGISTRY` (import by name);
      4. ``src.models.<name>``, picking its single ``SurrogateBase`` subclass.
    """
    name = get_in(cfg, "model.name")
    if not name:
        raise ModuleNotReadyError("config has no model.name")
    model_cfg = {k: v for k, v in dict(get_in(cfg, "model", {}) or {}).items() if k != "name"}
    key = str(name).lower()

    try:
        pkg = importlib.import_module("src.models")
    except Exception:
        pkg = None

    if pkg is not None:
        builder = getattr(pkg, "build_model", None)
        if callable(builder):
            try:
                return builder(name, model_cfg)
            except (KeyError, ValueError) as exc:
                raise ModuleNotReadyError(
                    f"src.models.build_model rejected model {name!r}: {exc}"
                ) from exc
        registry = getattr(pkg, "MODEL_REGISTRY", None)
        if isinstance(registry, Mapping) and key in {str(k).lower() for k in registry}:
            entry = next(v for k, v in registry.items() if str(k).lower() == key)
            cls = entry if isinstance(entry, type) else _import_target(entry)
            return cls(model_cfg)

    if key in FALLBACK_REGISTRY:
        return _import_target(FALLBACK_REGISTRY[key])

    try:  # last resort: a module named after the model with one SurrogateBase
        mod = importlib.import_module(f"src.models.{key}")
        from src.models.common import SurrogateBase

        cands = [
            v for v in vars(mod).values()
            if isinstance(v, type) and issubclass(v, SurrogateBase) and v is not SurrogateBase
        ]
        if len(cands) == 1:
            return cands[0](model_cfg)
    except Exception as exc:
        raise ModuleNotReadyError(
            f"cannot build model {name!r}: {type(exc).__name__}: {exc}. "
            f"Known names: {sorted(FALLBACK_REGISTRY)}"
        ) from exc
    raise ModuleNotReadyError(
        f"cannot resolve model {name!r}; known names: {sorted(FALLBACK_REGISTRY)}"
    )


def count_params(model) -> int | None:
    """Parameter count (all parameters, matching the 1.5 M budget of CONTEXT.md 7)."""
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def build_datasets(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build train/cal/test datasets via ``src.data.airfrans_loader``.

    Returns ``{'train': ds, 'cal': ds|None, 'test': ds|None, 'collate': fn}``.
    """
    try:
        loader_mod = importlib.import_module("src.data.airfrans_loader")
    except Exception as exc:
        raise ModuleNotReadyError(
            "src.data.airfrans_loader is not importable "
            f"({type(exc).__name__}: {exc}) -- run scripts/build_cache.py first "
            "and check that A1's data module landed."
        ) from exc

    Dataset = getattr(loader_mod, "AirfransSurfaceDataset", None)
    collate = getattr(loader_mod, "collate", None)
    if Dataset is None:
        raise ModuleNotReadyError(
            "src.data.airfrans_loader has no AirfransSurfaceDataset (CONTEXT.md 4)"
        )

    split = get_in(cfg, "data.split", "full")
    splits_dir = Path(get_in(cfg, "data.splits_dir", "data/splits"))
    split_file = get_in(cfg, "data.split_file") or splits_dir / f"{split}.json"
    processed = get_in(cfg, "data.processed_dir", "data/processed/airfrans")
    stats = get_in(cfg, "data.norm_stats", "auto")

    common = dict(
        split_file=split_file,
        processed_dir=processed,
        normalize_stats=stats if stats else "auto",
        load_volume=bool(get_in(cfg, "data.load_volume", False)),
        cache_in_ram=bool(get_in(cfg, "data.cache_in_ram", True)),
        missing=get_in(cfg, "data.missing", "error"),
    )
    out: dict[str, Any] = {"collate": collate}
    for subset in ("train", "cal", "test"):
        try:
            out[subset] = Dataset(subset=subset, **common)
        except Exception as exc:
            if subset == "train":
                raise ModuleNotReadyError(
                    f"cannot build the train dataset for split {split!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            out[subset] = None
    return out


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int | None = None,
                collate: Callable | None = None, num_workers: int = 0):
    """Deterministic DataLoader; ``seed`` drives the shuffling generator."""
    import torch
    from torch.utils.data import DataLoader

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": max(0, min(int(num_workers), 2)),  # CONTEXT.md 1
        "generator": generator,
    }
    if collate is not None:
        kwargs["collate_fn"] = collate
    return DataLoader(dataset, **kwargs)


# --------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg: Mapping[str, Any]):
    """Adam (CONTEXT.md 10); ``adamw`` is accepted for ablations."""
    import torch

    name = str(get_in(cfg, "train.optimizer", "adam")).lower()
    lr = float(get_in(cfg, "train.lr", 1e-3))
    wd = float(get_in(cfg, "train.weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name != "adam":
        raise ValueError(f"unsupported optimizer {name!r} (adam | adamw)")
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: Mapping[str, Any], epochs: int | None = None):
    """Cosine annealing over the full epoch budget (CONTEXT.md 10)."""
    import torch

    name = str(get_in(cfg, "train.scheduler", "cosine") or "none").lower()
    n = int(epochs if epochs is not None else get_in(cfg, "train.epochs", 400))
    if name in ("none", "constant"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n), eta_min=float(get_in(cfg, "train.min_lr", 1e-6))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(get_in(cfg, "train.step_size", max(1, n // 3))),
            gamma=float(get_in(cfg, "train.gamma", 0.1)),
        )
    raise ValueError(f"unsupported scheduler {name!r} (cosine | step | none)")


# --------------------------------------------------------------------------- #
# Loss (CONTEXT.md 8; fallback lives here until src/models/losses.py exists)
# --------------------------------------------------------------------------- #
def _seg_rel_l2(pred, true, batch_idx, n_graphs: int):
    """Per-simulation relative L2, averaged over the batch (torch, differentiable)."""
    import torch

    diff2 = (pred - true) ** 2
    true2 = true**2
    if diff2.dim() > 1:
        diff2 = diff2.sum(dim=tuple(range(1, diff2.dim())))
        true2 = true2.sum(dim=tuple(range(1, true2.dim())))
    num = torch.zeros(n_graphs, device=pred.device, dtype=pred.dtype)
    den = torch.zeros(n_graphs, device=pred.device, dtype=pred.dtype)
    num.index_add_(0, batch_idx, diff2)
    den.index_add_(0, batch_idx, true2)
    return (num.clamp_min(0).sqrt() / (den.clamp_min(0).sqrt() + EPS)).mean()


def fallback_total_loss(
    pred: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    norm_ref: dict[str, float] | None = None,
    *,
    denorm: Callable[[str, Any], Any] | None = None,
    model=None,
    integrate: Callable | None = None,
) -> tuple[Any, dict[str, float]]:
    """CONTEXT.md 8 ``total_loss`` semantics, implemented locally.

    ``L = L_data + lambda_F * L_force + lambda_S * L_sym`` with
    ``L_data`` = relative L2 on p and tau (normalized space) and each auxiliary
    term divided by its value on the first batch (``norm_ref``, mutated in
    place).  Defaults: every lambda 0, i.e. pure data loss.

    Used until ``src/models/losses.py`` exists; :func:`resolve_loss_fn` prefers
    that module the moment it appears.
    """
    import torch

    w = dict(weights or {})
    norm_ref = norm_ref if norm_ref is not None else {}

    bidx = batch["batch_idx"]
    n_graphs = int(batch.get("num_graphs") or (int(bidx.max().item()) + 1))

    p_loss = _seg_rel_l2(pred["p"].reshape(-1), batch["surf_p"].reshape(-1), bidx, n_graphs)
    tau_loss = _seg_rel_l2(pred["tau"], batch["surf_tau"], bidx, n_graphs)
    total = p_loss + tau_loss
    parts = {
        "loss": float(total.detach()),
        "p_rel_l2": float(p_loss.detach()),
        "tau_rel_l2": float(tau_loss.detach()),
        "force": math.nan,
        "sym": math.nan,
        "head": math.nan,
    }

    lam_f = float(w.get("force", 0.0) or 0.0)
    if lam_f > 0.0:
        force = _force_term(pred, batch, denorm, integrate, n_graphs)
        if force is not None:
            ref = norm_ref.setdefault("force", max(float(force.detach()), EPS))
            total = total + lam_f * force / ref
            parts["force"] = float(force.detach())

    lam_h = float(w.get("head", DEFAULT_LAMBDA_H) or 0.0)  # D-016
    if lam_h > 0.0 and "cl_true" in batch and "cd_true" in batch:
        head = head_regression_loss(pred, batch)
        ref = norm_ref.setdefault("head", max(float(head.detach()), EPS))
        total = total + lam_h * head / ref
        parts["head"] = float(head.detach())

    lam_s = float(w.get("sym", 0.0) or 0.0)
    if lam_s > 0.0 and model is not None:
        sym = _sym_term(model, batch, pred)
        if sym is not None:
            ref = norm_ref.setdefault("sym", max(float(sym.detach()), EPS))
            total = total + lam_s * sym / ref
            parts["sym"] = float(sym.detach())

    parts["loss"] = float(total.detach())
    return total, parts


def _force_term(pred, batch, denorm, integrate, n_graphs):
    """``|C_int - C_true| + |C_head - C_true| + |C_int - C_head|`` (spec 5.4)."""
    import torch

    if integrate is None:
        integrate = _resolve_integrate()
    if integrate is None or denorm is None:
        return None
    try:
        p = denorm("surf_p", pred["p"].reshape(-1))
        tau = denorm("surf_tau", pred["tau"])
        cond = batch.get("cond_phys")
        if cond is None:
            cond = denorm("cond", batch["cond"])
        forces = integrate(
            p, tau, batch["surf_normal"], batch["surf_ds"], cond,
            batch_idx=batch["batch_idx"],
        )
        c_int = torch.stack([forces["cl"].reshape(-1), forces["cd"].reshape(-1)], dim=1)
        coef = pred["coef_head"].reshape(n_graphs, 2)
        c_head = torch.stack(
            [denorm("cl_true", coef[:, 0]), denorm("cd_true", coef[:, 1])], dim=1
        )
        cl_t = batch.get("cl_true_phys")
        cd_t = batch.get("cd_true_phys")
        if cl_t is None or cd_t is None:
            cl_t = denorm("cl_true", batch["cl_true"])
            cd_t = denorm("cd_true", batch["cd_true"])
        c_true = torch.stack([cl_t.reshape(-1), cd_t.reshape(-1)], dim=1)
        return (
            (c_int - c_true).abs().mean()
            + (c_head - c_true).abs().mean()
            + (c_int - c_head).abs().mean()
        )
    except Exception:
        return None


def _sym_term(model, batch, pred):
    """``||u - R# u(Rg,Rc)|| / ||u||`` (spec 5.5), one extra forward pass."""
    import torch

    from src.eval.harness import _reflect  # local: torch-dependent

    try:
        pred_r = model(_reflect(batch))
        tau_r = pred_r["tau"].clone()
        tau_r = torch.stack([tau_r[:, 0], -tau_r[:, 1]], dim=1)
        u = torch.cat([pred["p"].reshape(-1), pred["tau"].reshape(-1)])
        u_r = torch.cat([pred_r["p"].reshape(-1), tau_r.reshape(-1)])
        return torch.linalg.vector_norm(u - u_r) / (torch.linalg.vector_norm(u) + EPS)
    except Exception:
        return None


def _resolve_integrate():
    try:
        mod = importlib.import_module("src.physics.force_integration")
        return getattr(mod, "integrate_forces", None)
    except Exception:
        return None


#: D-016: the coefficient-head regression weight is part of the base objective.
DEFAULT_LAMBDA_H = 0.1

#: ``src.models.losses`` term names -> our ``history.csv`` column names.
_TERM_ALIASES = {"data_p": "p_rel_l2", "data_tau": "tau_rel_l2",
                 "force": "force", "sym": "sym", "head": "head", "loss": "loss"}


def _f(value) -> float:
    try:
        return float(value.detach() if hasattr(value, "detach") else value)
    except Exception:  # pragma: no cover - defensive
        return math.nan


def make_denorm_integrator(denorm: Callable | None, renorm: Callable | None = None,
                           integrate: Callable | None = None):
    """Wrap ``integrate_forces`` so units line up on both sides of the loss.

    ``src.models.losses`` deliberately knows nothing about normalisation and
    passes the model's normalized predictions plus the batch's ``cond``
    straight through (its module docstring asks the trainer to supply exactly
    this closure).  So we

    1. **denormalize** p, tau and cond -- forces are only meaningful in
       physical units (D-014: positions, normals and ``ds`` are already
       physical, only p/tau/cond are standardized); then
    2. **re-normalize** the resulting CL/CD, because the loss compares them
       against ``coef_head`` and ``cl_true``/``cd_true``, which are in
       normalized space.  Skipping this silently mixes units in
       ``force_consistency_loss``.

    Both transforms are affine, so gradients flow through unchanged.
    """
    integrate = integrate or _resolve_integrate()
    if integrate is None:
        return None

    def wrapped(p, tau, normal, ds, cond, batch_idx=None, **kw):
        if denorm is not None:
            p = denorm("surf_p", p)
            tau = denorm("surf_tau", tau)
            cond = denorm("cond", cond)
        out = dict(integrate(p, tau, normal, ds, cond, batch_idx=batch_idx, **kw))
        if renorm is not None:
            out["cl"] = renorm("cl_true", out["cl"])
            out["cd"] = renorm("cd_true", out["cd"])
        return out

    return wrapped


def head_regression_loss(pred: Mapping[str, Any], batch: Mapping[str, Any]):
    """``|CL_head - CL_true| + |CD_head - CD_true|`` (D-016, weight ``lambda_H``).

    Without this term the coefficient head receives **no gradient at all** under
    the pure field loss, which makes ``coef_head`` -- and therefore every FSC
    number in ``metrics.json`` -- meaningless.  It is part of the base
    objective, not an ablation: ``lambda_F`` stays the separate
    force-*consistency* term.

    Computed in normalized space, like the rest of ``src.models.losses``: the
    head predicts normalized coefficients and the batch carries normalized
    ``cl_true``/``cd_true``, so the two are directly comparable and the term is
    scaled to order one anyway by ``norm_ref``.
    """
    import torch

    coef = pred["coef_head"]
    n = coef.shape[0]
    target = torch.stack(
        [batch["cl_true"].reshape(n), batch["cd_true"].reshape(n)], dim=-1
    ).to(coef.dtype)
    return (coef.reshape(n, 2) - target).abs().sum(dim=-1).mean()


def make_symmetry_fn(model):
    """``symmetry_fn(pred, batch) -> scalar`` for ``src.models.losses`` (spec 5.5)."""
    if model is None:
        return None

    def fn(pred, batch):
        value = _sym_term(model, batch, pred)
        if value is None:  # pragma: no cover - reflection unavailable
            import torch

            return torch.zeros((), device=pred["p"].device, dtype=pred["p"].dtype)
        return value

    return fn


def resolve_loss_fn(cfg: Mapping[str, Any], *, denorm=None, renorm=None,
                    model=None) -> Callable:
    """Return ``fn(pred, batch, norm_ref) -> (loss, parts)``.

    Prefers ``src.models.losses.total_loss`` (CONTEXT.md 8), injecting the
    unit-correcting force integrator, the symmetry hook, and the D-016
    coefficient-head term (``extra_fns['head']``), and capturing ``norm_ref``
    on the first batch via ``init_norm_ref``.  Falls back to
    :func:`fallback_total_loss` when that module is absent.
    """
    weights = dict(get_in(cfg, "train.weights", {}) or {})
    weights.setdefault("head", DEFAULT_LAMBDA_H)  # D-016: always on
    lam_f = float(weights.get("force", 0.0) or 0.0)
    lam_s = float(weights.get("sym", 0.0) or 0.0)
    lam_h = float(weights.get("head", 0.0) or 0.0)

    try:
        losses = importlib.import_module("src.models.losses")
    except Exception:
        losses = None
    total_loss = getattr(losses, "total_loss", None) if losses else None

    if callable(total_loss):
        integrate_fn = make_denorm_integrator(denorm, renorm=renorm) if lam_f else None
        symmetry_fn = make_symmetry_fn(model) if lam_s else None
        extra_fns = {"head": head_regression_loss} if lam_h else None
        init_norm_ref = getattr(losses, "init_norm_ref", None)
        needs_ref = bool(lam_f or lam_s or lam_h)

        def fn(pred, batch, norm_ref):
            if needs_ref and callable(init_norm_ref) and norm_ref is not None and not norm_ref:
                norm_ref.update(
                    init_norm_ref(pred, batch, weights, integrate_fn, symmetry_fn,
                                  extra_fns)
                )
            loss, terms = total_loss(
                pred, batch, weights, norm_ref,
                integrate_fn=integrate_fn, symmetry_fn=symmetry_fn,
                extra_fns=extra_fns,
            )
            parts = {"loss": _f(loss), "p_rel_l2": math.nan, "tau_rel_l2": math.nan,
                     "force": math.nan, "sym": math.nan}
            for src_key, dst_key in _TERM_ALIASES.items():
                if src_key in terms:
                    parts[dst_key] = _f(terms[src_key])
            return loss, parts

        return fn

    integrate = _resolve_integrate()

    def fn(pred, batch, norm_ref):
        return fallback_total_loss(
            pred, batch, weights, norm_ref,
            denorm=denorm, model=model, integrate=integrate,
        )

    return fn


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path, *, model, optimizer=None, scheduler=None, epoch: int,
                    best_metric: float | None = None, config: Mapping | None = None,
                    norm_ref: Mapping | None = None, extra: Mapping | None = None):
    """Write a full checkpoint atomically (model+optim+sched+epoch+RNG+norm_ref).

    ``norm_ref`` (the auxiliary-loss scales captured at init, CONTEXT.md 8) is
    part of the objective, so it must survive an interruption: re-capturing it
    on the first batch of a *resumed* session would silently change the loss
    and break bit-identical resume.
    """
    import torch

    from src.utils.io import ensure_dir

    path = Path(path)
    ensure_dir(path.parent)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": None if best_metric is None else float(best_metric),
        "norm_ref": dict(norm_ref) if norm_ref is not None else None,
        "rng": get_rng_state(),
        "config": dict(config) if config is not None else None,
        "torch_version": torch.__version__,
    }
    if extra:
        payload.update(dict(extra))
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leave a truncated last.pt behind
    return path


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, *,
                    map_location="cpu", restore_rng: bool = True) -> dict:
    """Load a checkpoint, restoring model/optimizer/scheduler/RNG in place."""
    import torch

    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    if model is not None and ckpt.get("model") is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        set_rng_state(ckpt.get("rng"))
    return ckpt


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
def _to_device(batch, device):
    from src.eval.harness import batch_to

    return batch_to(batch, device)


def validate(model, dataset, cfg, *, device, collate=None, denorm=None,
             loss_fn=None, limit_batches: int | None = None,
             norm_ref: dict | None = None) -> dict[str, float]:
    """One validation pass: loss + field rel-L2 + coefficient MAE (physical)."""
    import torch

    from src.eval.harness import _phys

    if dataset is None or len(dataset) == 0:
        return {}
    loader = make_loader(
        dataset,
        batch_size=int(get_in(cfg, "eval.batch_size", get_in(cfg, "train.batch_size", 8))),
        shuffle=False,
        collate=collate,
        num_workers=0,
    )
    model.eval()
    tot = {"val_loss": 0.0, "val_p_rel_l2": 0.0, "val_tau_rel_l2": 0.0}
    cl_err: list[float] = []
    cd_err: list[float] = []
    nb = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            batch = _to_device(batch, device)
            pred = model(batch)
            if loss_fn is not None:
                # reuse the training norm_ref: the auxiliary terms must be
                # scaled by their value at init, not re-captured on val data
                _, parts = loss_fn(pred, batch, norm_ref if norm_ref is not None else {})
                tot["val_loss"] += float(parts.get("loss", math.nan))
                tot["val_p_rel_l2"] += float(parts.get("p_rel_l2", math.nan))
                tot["val_tau_rel_l2"] += float(parts.get("tau_rel_l2", math.nan))
            nb += 1
            if denorm is not None and "coef_head" in pred:
                n = int(batch.get("num_graphs") or pred["coef_head"].shape[0])
                coef = pred["coef_head"].reshape(n, 2)
                cl_p = denorm("cl_true", coef[:, 0])
                cd_p = denorm("cd_true", coef[:, 1])
                cl_t = _phys(batch, "cl_true", denorm).reshape(-1)
                cd_t = _phys(batch, "cd_true", denorm).reshape(-1)
                cl_err += (cl_p - cl_t).abs().detach().cpu().tolist()
                cd_err += (cd_p - cd_t).abs().detach().cpu().tolist()
    model.train()
    out = {k: (v / nb if nb else math.nan) for k, v in tot.items()}
    out["val_cl_mae"] = float(sum(cl_err) / len(cl_err)) if cl_err else math.nan
    out["val_cd_mae"] = float(sum(cd_err) / len(cd_err)) if cd_err else math.nan
    return out


def train_model(
    model,
    train_dataset,
    val_dataset,
    cfg: Mapping[str, Any],
    *,
    ckpt_dir,
    history_path=None,
    device="cpu",
    collate: Callable | None = None,
    denorm: Callable | None = None,
    renorm: Callable | None = None,
    loss_fn: Callable | None = None,
    limit_batches: int | None = None,
    resume: bool = True,
    stop_after_epochs: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the training loop with per-epoch checkpointing and auto-resume.

    ``stop_after_epochs`` bounds how many epochs *this call* runs (the total
    budget stays ``train.epochs``, so the cosine schedule is unchanged).  It
    models an interrupted session -- a killed Kaggle kernel, a closed lid --
    and is what ``tests/test_infra.py`` uses to check that resuming reproduces
    uninterrupted training bit for bit.

    Returns ``{'epochs': n, 'train_time_s': s, 'best_metric': m, 'history': [...]}``.
    """
    import torch

    from src.utils.io import ensure_dir

    device = torch.device(device)
    model.to(device)
    model.train()

    ckpt_dir = ensure_dir(ckpt_dir)
    last_path = Path(ckpt_dir) / "last.pt"
    best_path = Path(ckpt_dir) / "best.pt"

    epochs = int(get_in(cfg, "train.epochs", 400))
    batch_size = int(get_in(cfg, "train.batch_size", 16))
    grad_clip = float(get_in(cfg, "train.grad_clip", 1.0) or 0.0)
    seed = int(get_in(cfg, "seed", 0))
    mode = str(get_in(cfg, "train.val_metric_mode", "min")).lower()
    metric_key = str(get_in(cfg, "train.val_metric", "val_loss"))
    num_workers = int(get_in(cfg, "train.num_workers", 0))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)
    if loss_fn is None:
        loss_fn = resolve_loss_fn(cfg, denorm=denorm, renorm=renorm, model=model)

    start_epoch = 0
    best_metric = math.inf if mode == "min" else -math.inf
    resume_ckpt = None
    if resume and last_path.is_file():
        resume_ckpt = load_checkpoint(last_path, model, optimizer, scheduler,
                                      map_location=device)
        start_epoch = int(resume_ckpt.get("epoch", 0))
        if resume_ckpt.get("best_metric") is not None:
            best_metric = float(resume_ckpt["best_metric"])
        log(f"[train] resumed from {last_path} at epoch {start_epoch}")

    norm_ref: dict[str, float] = {}
    if resume_ckpt is not None and resume_ckpt.get("norm_ref"):
        norm_ref.update(resume_ckpt["norm_ref"])
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        loader = make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed * 1_000_003 + epoch,  # depends on epoch only -> resume-safe
            collate=collate,
            num_workers=num_workers,
        )
        lr = float(optimizer.param_groups[0]["lr"])
        sums = {"loss": 0.0, "p_rel_l2": 0.0, "tau_rel_l2": 0.0,
                "head": 0.0, "force": 0.0, "sym": 0.0}
        nb = 0
        model.train()
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss, parts = loss_fn(pred, batch, norm_ref)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch} batch {i}: {float(loss)}"
                )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for k in sums:
                v = parts.get(k, math.nan)
                sums[k] += 0.0 if (v is None or not math.isfinite(v)) else float(v)
            nb += 1
        if scheduler is not None:
            scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sums["loss"] / nb if nb else math.nan,
            "train_p_rel_l2": sums["p_rel_l2"] / nb if nb else math.nan,
            "train_tau_rel_l2": sums["tau_rel_l2"] / nb if nb else math.nan,
            "train_head": sums["head"] / nb if nb else math.nan,
            "train_force": sums["force"] / nb if nb else math.nan,
            "train_sym": sums["sym"] / nb if nb else math.nan,
            "lr": lr,
        }
        every = max(1, int(get_in(cfg, "train.val_every", 1)))
        if val_dataset is not None and (epoch % every == 0 or epoch == epochs - 1):
            row.update(
                validate(model, val_dataset, cfg, device=device, collate=collate,
                         denorm=denorm, loss_fn=loss_fn, limit_batches=limit_batches,
                         norm_ref=norm_ref)
            )
        row["epoch_time_s"] = time.perf_counter() - t0
        history.append(row)
        if history_path is not None:
            append_csv_row(history_path, row, HISTORY_FIELDS)

        metric = row.get(metric_key, row.get("train_loss", math.nan))
        improved = (
            metric is not None
            and isinstance(metric, float)
            and math.isfinite(metric)
            and ((metric < best_metric) if mode == "min" else (metric > best_metric))
        )
        if improved:
            best_metric = float(metric)

        save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch + 1, best_metric=best_metric, config=cfg,
                        norm_ref=norm_ref)
        if improved:
            save_checkpoint(best_path, model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch + 1,
                            best_metric=best_metric, config=cfg, norm_ref=norm_ref)

        log(
            f"[train] epoch {epoch + 1}/{epochs} "
            f"loss={row['train_loss']:.5f} "
            f"{metric_key}={row.get(metric_key, float('nan'))} "
            f"lr={lr:.3e} {row['epoch_time_s']:.1f}s"
        )

        if stop_after_epochs is not None and len(history) >= stop_after_epochs:
            log(f"[train] stopping after {len(history)} epoch(s) this session; "
                f"last.pt at epoch {epoch + 1} will resume")
            break

    return {
        "epochs": epochs,
        "epochs_run": epochs - start_epoch,
        "train_time_s": time.perf_counter() - t_start,
        "best_metric": None if not math.isfinite(best_metric) else best_metric,
        "history": history,
        "last_path": last_path,
        "best_path": best_path if best_path.is_file() else last_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/train.py",
        description="Train a surrogate; checkpoints every epoch and auto-resumes.",
        epilog="Overrides are dotted key=value pairs, e.g. split=full seed=0 train.lr=3e-4",
    )
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/gnn.yaml")
    p.add_argument("overrides", nargs="*", default=[], help="dotted key=value overrides")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, 2 batches each, tag suffixed '_dryrun' so real "
                        "results are never overwritten")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="cap batches per epoch (smoke tests)")
    p.add_argument("--device", default=None, help="cpu | cuda | auto (default: config)")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    p.add_argument("--no-eval", action="store_true", help="skip the final eval harness")
    p.add_argument("--results-root", default=None)
    p.add_argument("--checkpoints-root", default=None)
    return p


def parse_cli(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None):
    """Parse args, accepting ``key=value`` overrides **anywhere** on the line.

    ``argparse`` fills a ``nargs="*"`` positional greedily at its first
    opportunity, so ``--checkpoint X tag=eval --device cpu split=aoa`` would
    otherwise fail on the trailing overrides.  Everything unrecognised that
    contains ``=`` and does not start with ``-`` is collected as an override;
    anything else is still a genuine CLI error.
    """
    args, unknown = parser.parse_known_args(argv)
    extra, bad = [], []
    for item in unknown:
        (extra if ("=" in item and not item.startswith("-")) else bad).append(item)
    if bad:
        parser.error("unrecognized arguments: " + " ".join(bad))
    args.overrides = list(getattr(args, "overrides", None) or []) + extra
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Load config, train, then produce ``results/<run_id>/metrics.json``."""
    args = parse_cli(build_parser(), argv)
    cfg = load_config(args.config, args.overrides)

    if args.device:
        cfg["device"] = args.device
    if args.results_root:
        cfg.setdefault("paths", {})["results_root"] = args.results_root
    if args.checkpoints_root:
        cfg.setdefault("paths", {})["checkpoints_root"] = args.checkpoints_root

    limit = args.limit_batches
    if args.dry_run:
        cfg["train"]["epochs"] = min(2, int(get_in(cfg, "train.epochs", 2)))
        limit = limit or 2
        tag = cfg.get("tag")
        cfg["tag"] = f"{tag}_dryrun" if tag else "dryrun"

    seed = int(get_in(cfg, "seed", 0))
    seed_all(seed)
    device = resolve_device(cfg.get("device", "auto"))

    run_id = build_run_id(
        get_in(cfg, "model.name"), get_in(cfg, "data.split"), seed, cfg.get("tag")
    )
    rdir = results_dir(run_id, get_in(cfg, "paths.results_root", "results"))
    cdir = checkpoint_dir(run_id, get_in(cfg, "paths.checkpoints_root", "checkpoints"))
    cfg.setdefault("_meta", {})["run_id"] = run_id
    save_config(cfg, rdir / "config.yaml")
    print(f"[train] run_id={run_id} device={device} results={rdir} ckpt={cdir}")

    data = build_datasets(cfg)
    train_ds, cal_ds, test_ds = data["train"], data.get("cal"), data.get("test")
    collate = data.get("collate")
    denorm = getattr(train_ds, "denormalize", None)
    renorm = getattr(train_ds, "normalize", None)

    model = build_model(cfg)
    n_params = count_params(model)
    print(f"[train] model={get_in(cfg, 'model.name')} params={n_params:,}"
          f" train={len(train_ds)} cal={len(cal_ds) if cal_ds else 0}"
          f" test={len(test_ds) if test_ds else 0}")

    result = train_model(
        model, train_ds, cal_ds, cfg,
        ckpt_dir=cdir,
        history_path=rdir / "history.csv",
        device=device,
        collate=collate,
        denorm=denorm,
        renorm=renorm,
        limit_batches=limit,
        resume=not args.no_resume,
    )
    print(f"[train] done in {result['train_time_s']:.1f}s "
          f"best {get_in(cfg, 'train.val_metric', 'val_loss')}={result['best_metric']}")

    if args.no_eval:
        return 0

    from src.eval.harness import evaluate

    eval_ds = test_ds if test_ds is not None and len(test_ds) else cal_ds
    if eval_ds is None or len(eval_ds) == 0:
        print("[train] no test/cal split available; skipping metrics.json")
        return 0

    # Evaluate the best checkpoint, not the last one.
    best = result["best_path"]
    if Path(best).is_file():
        load_checkpoint(best, model, map_location=device, restore_rng=False)

    metrics = evaluate(
        model, eval_ds, str(get_in(cfg, "data.split", "full")),
        run_meta={
            "run_id": run_id,
            "model": str(get_in(cfg, "model.name")),
            "seed": seed,
            "tag": cfg.get("tag"),
            "params": n_params,
            "train_time_s": result["train_time_s"],
            "epochs": result["epochs"],
        },
        device=device,
        batch_size=int(get_in(cfg, "eval.batch_size", 8)),
        collate_fn=collate,
        max_batches=limit,
        compute_symmetry=bool(get_in(cfg, "eval.compute_symmetry", True)),
        symmetry_max_sims=get_in(cfg, "eval.symmetry_max_sims", 64),
        save_dir=rdir,
    )
    print(f"[train] metrics.json written: "
          f"p_rel_l2={metrics['field']['p_rel_l2']} "
          f"cd_int_mae={metrics['coef']['cd_int_mae']} "
          f"cd_spearman={metrics['coef']['cd_spearman']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
