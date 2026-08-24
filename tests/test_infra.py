"""Infrastructure tests (A4): config, seeding, IO, metrics, schema, train/resume.

Everything here runs on **CPU** and uses only ``src/utils``, ``src/eval`` and
``scripts/`` plus the synthetic stand-ins defined below -- no ``src/data``,
``src/models`` or ``src/physics``.  ``TinyModel`` implements the frozen model
interface of CONTEXT.md 7 and ``tiny_integrate`` the force-integration
signature of CONTEXT.md 6, so these tests also serve as executable
documentation of what the harness expects from the other agents' modules.

The headline test is :func:`test_checkpoint_resume_is_bit_identical`: two
epochs straight must equal one epoch + resume + one epoch, parameter for
parameter.  Per-epoch checkpointing is what makes interruptions free on this
machine (CONTEXT.md 10), and it is only worth anything if resume is exact.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sweep as sweep_mod  # noqa: E402
from scripts import train as train_mod  # noqa: E402
from src.eval import baselines as baselines_mod  # noqa: E402
from src.eval.harness import evaluate, reflect_batch  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    MetricAccumulator,
    coef_metrics,
    fsc,
    mae,
    rel_l2,
    spearman,
)
from src.eval.schema import (  # noqa: E402
    SchemaError,
    assert_valid_metrics,
    check_metrics,
    check_uq,
    empty_metrics,
    validate_metrics,
    validate_uq,
)
from src.utils.config import (  # noqa: E402
    ConfigError,
    get_in,
    infer_type,
    load_config,
    parse_override,
    save_config,
)
from src.utils.io import (  # noqa: E402
    append_csv_row,
    build_run_id,
    read_csv,
    read_json,
    write_json,
)
from src.utils.seed import get_rng_state, seed_all, set_rng_state  # noqa: E402


# =========================================================================== #
# Synthetic stand-ins for A1 (data), A2 (physics) and A3 (models)
# =========================================================================== #
def make_item(idx: int, n_pts: int = 24) -> dict:
    """One simulation on a circle, in the CONTEXT.md 4 item schema.

    Includes the ``*_phys`` copies A1's loader emits, so the harness can take
    ground truth in physical units without inverting a normalisation.
    """
    theta = torch.linspace(0.0, 2 * math.pi, n_pts + 1)[:-1] + 0.01 * idx
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).float()
    normal = pos.clone()  # outward unit normals on the unit circle
    ds = torch.full((n_pts,), float(2 * math.pi / n_pts))
    p = (0.3 * torch.cos(theta) + 0.05 * idx).float()
    tau = torch.stack([0.01 * torch.sin(theta), 0.01 * torch.cos(theta)], dim=1).float()
    aoa = math.radians(-4.0 + idx)
    cond = torch.tensor([50.0 * math.cos(aoa), 50.0 * math.sin(aoa)]).float()
    item = {
        "sim_name": f"airFoil2D_SST_{50.0 + idx:.3f}_{-4.0 + idx:.3f}_"
                    f"{1.0 + 0.1 * idx:.3f}_{4.0:.3f}_{12.0 + idx:.3f}",
        "surf_pos": pos,
        "surf_normal": normal,
        "surf_ds": ds,
        "surf_p": p,
        "surf_tau": tau,
        "cond": cond,
        "cl_true": torch.tensor(0.1 * idx - 0.2),
        "cd_true": torch.tensor(0.02 + 0.001 * idx),
        "n_surf": torch.tensor(n_pts, dtype=torch.long),
    }
    for key in ("surf_p", "surf_tau", "cond", "cl_true", "cd_true"):
        item[f"{key}_phys"] = item[key].clone()
    return item


class TinyDataset(torch.utils.data.Dataset):
    """Map-style dataset of synthetic sims; no files, no normalisation."""

    def __init__(self, n: int = 6, n_pts: int = 24):
        self.items = [make_item(i, n_pts) for i in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def tiny_collate(items) -> dict:
    """PyG-style collate: concatenate points, stack per-sim, add ``batch_idx``."""
    batch: dict = {}
    for key in items[0]:
        if key.startswith("surf_"):
            batch[key] = torch.cat([it[key] for it in items], dim=0)
    counts = torch.tensor([int(it["n_surf"]) for it in items], dtype=torch.long)
    batch["n_surf"] = counts
    batch["batch_idx"] = torch.repeat_interleave(torch.arange(len(items)), counts)
    batch["ptr"] = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    for key in ("cond", "cond_phys", "cl_true", "cl_true_phys",
                "cd_true", "cd_true_phys"):
        batch[key] = torch.stack([it[key] for it in items], dim=0)
    batch["sim_name"] = [it["sim_name"] for it in items]
    batch["num_graphs"] = len(items)
    return batch


class TinyModel(nn.Module):
    """Minimal model satisfying CONTEXT.md 7.

    ``forward(batch) -> {'p': (sumN,), 'tau': (sumN,2), 'coef_head': (B,2)}``.
    Point features are (pos, normal, broadcast cond); the coefficient head is a
    ds-weighted mean pool, mirroring the shared head of ``src/models/heads.py``.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.p_head = nn.Linear(hidden, 1)
        self.tau_head = nn.Linear(hidden, 2)
        self.coef_head_mlp = nn.Linear(hidden, 2)

    def forward(self, batch: dict) -> dict:
        pos, normal = batch["surf_pos"], batch["surf_normal"]
        bidx = batch["batch_idx"]
        cond = batch["cond"][bidx]
        h = self.trunk(torch.cat([pos, normal, cond * 0.02], dim=1))
        n_graphs = int(batch.get("num_graphs") or int(bidx.max()) + 1)
        pooled = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        counts = torch.zeros(n_graphs, dtype=h.dtype, device=h.device)
        pooled.index_add_(0, bidx, h)
        counts.index_add_(0, bidx, torch.ones_like(bidx, dtype=h.dtype))
        pooled = pooled / counts.clamp_min(1).unsqueeze(1)
        return {
            "p": self.p_head(h).squeeze(-1),
            "tau": self.tau_head(h),
            "coef_head": self.coef_head_mlp(pooled),
        }


def tiny_integrate(p, tau, normal, ds, cond, batch_idx=None, rho=1.0, a_ref=1.0):
    """Stand-in for CONTEXT.md 6 ``integrate_forces`` (same signature/convention).

    ``F = sum (-p n + tau) ds``; drag along ``e_inf``, lift along ``rot90(e_inf)``,
    both divided by ``0.5 rho |u|^2 A_ref``.
    """
    if batch_idx is None:
        batch_idx = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
    cond = cond.reshape(-1, 2)
    b = cond.shape[0]
    contrib = (-p.unsqueeze(1) * normal + tau) * ds.unsqueeze(1)
    force = torch.zeros(b, 2, dtype=contrib.dtype, device=contrib.device)
    force.index_add_(0, batch_idx, contrib)
    u = torch.linalg.vector_norm(cond, dim=1, keepdim=True).clamp_min(1e-12)
    e_inf = cond / u
    e_perp = torch.stack([-e_inf[:, 1], e_inf[:, 0]], dim=1)
    q = 0.5 * rho * (u.squeeze(1) ** 2) * a_ref
    return {
        "cd": (force * e_inf).sum(1) / q,
        "cl": (force * e_perp).sum(1) / q,
        "fx": force[:, 0],
        "fy": force[:, 1],
    }


def tiny_cfg(**over) -> dict:
    """A resolved-config-shaped dict for the trainer."""
    cfg = {
        "seed": 0,
        "tag": None,
        "device": "cpu",
        "model": {"name": "tiny"},
        "train": {
            "epochs": 2, "lr": 1e-2, "batch_size": 2, "weight_decay": 0.0,
            "optimizer": "adam", "scheduler": "cosine", "min_lr": 1e-6,
            "grad_clip": 1.0, "num_workers": 0, "val_every": 1,
            "val_metric": "val_loss", "val_metric_mode": "min",
            "weights": {"force": 0.0, "sym": 0.0, "bc": 0.0, "div": 0.0},
        },
        "data": {"split": "full"},
        "eval": {"batch_size": 2},
    }
    for dotted, value in over.items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


# =========================================================================== #
# 1. Config: overrides and type inference
# =========================================================================== #
def test_infer_type_covers_the_awkward_cases():
    # PyYAML's 1.1 resolver leaves '3e-4' as a string -- the whole reason
    # infer_type exists rather than a bare yaml.safe_load.
    assert infer_type("3e-4") == pytest.approx(3e-4)
    assert isinstance(infer_type("3e-4"), float)
    assert infer_type("400") == 400 and isinstance(infer_type("400"), int)
    assert infer_type("1.0") == pytest.approx(1.0)
    assert infer_type("-2.5e2") == pytest.approx(-250.0)
    assert infer_type("true") is True and infer_type("False") is False
    assert infer_type("null") is None
    assert infer_type("full") == "full"
    assert infer_type("[1, 2, 3]") == [1, 2, 3]
    assert infer_type("a,b") == ["a", "b"]
    assert infer_type("'3e-4'") == "3e-4"  # quoting forces a string


def test_parse_override_expands_bare_aliases():
    assert parse_override("train.lr=3e-4") == ("train.lr", pytest.approx(3e-4))
    assert parse_override("split=full") == ("data.split", "full")
    assert parse_override("seed=3") == ("seed", 3)
    assert parse_override("data.split=aoa") == ("data.split", "aoa")
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_load_config_applies_overrides_and_types(tmp_path):
    cfg_file = tmp_path / "m.yaml"
    cfg_file.write_text(
        "model:\n  name: gnn\n  k: 16\ntrain:\n  lr: 0.001\n  epochs: 400\n"
        "data:\n  split: full\n",
        encoding="utf-8",
    )
    cfg = load_config(
        cfg_file,
        ["train.lr=3e-4", "split=aoa", "seed=7", "train.epochs=50",
         "model.layer_norm=false", "train.weights.force=0.1", "new.nested.key=5"],
    )
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["data"]["split"] == "aoa"
    assert cfg["seed"] == 7
    assert cfg["train"]["epochs"] == 50
    assert cfg["model"]["layer_norm"] is False
    assert cfg["model"]["k"] == 16          # untouched file value survives
    assert cfg["train"]["batch_size"] == 16  # default merged in
    assert cfg["train"]["weights"]["force"] == pytest.approx(0.1)
    assert get_in(cfg, "new.nested.key") == 5
    assert cfg["_meta"]["overrides"][0] == "train.lr=3e-4"


def test_load_config_rejects_a_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_save_config_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "gnn.yaml", ["split=scarce", "seed=2"])
    out = save_config(cfg, tmp_path / "run" / "config.yaml")
    assert out.is_file()
    again = load_config(out)
    assert again["data"]["split"] == "scarce"
    assert again["seed"] == 2
    assert again["model"]["name"] == "gnn"


@pytest.mark.parametrize("name", ["gnn", "sdf_fno", "transolver"])
def test_shipped_configs_have_the_frozen_defaults(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["model"]["name"] == name
    assert cfg["train"]["epochs"] == 400            # CONTEXT.md 10
    assert cfg["train"]["lr"] == pytest.approx(1e-3)
    assert cfg["train"]["batch_size"] == 16
    assert cfg["train"]["grad_clip"] == pytest.approx(1.0)
    assert cfg["train"]["scheduler"] == "cosine"
    assert cfg["train"]["num_workers"] <= 2          # CONTEXT.md 1
    assert set(cfg["train"]["weights"]) >= {"head", "force", "sym"}
    # D-016: the coefficient head must always get gradient, or FSC is noise
    assert cfg["train"]["weights"]["head"] == pytest.approx(0.1)
    assert cfg["train"]["weights"]["force"] == 0.0   # ablation axis, off by default
    assert cfg["train"]["weights"]["sym"] == 0.0


# =========================================================================== #
# 2. Seeding
# =========================================================================== #
def test_seed_all_makes_python_numpy_torch_reproducible():
    seed_all(1234)
    a = (random.random(), np.random.rand(3).tolist(), torch.randn(4).tolist())
    seed_all(1234)
    b = (random.random(), np.random.rand(3).tolist(), torch.randn(4).tolist())
    assert a == b

    seed_all(1235)
    c = (random.random(), np.random.rand(3).tolist(), torch.randn(4).tolist())
    assert a != c


def test_rng_state_roundtrip_restores_the_stream():
    seed_all(0)
    state = get_rng_state()
    first = (random.random(), np.random.rand(2).tolist(), torch.randn(3).tolist())
    set_rng_state(state)
    second = (random.random(), np.random.rand(2).tolist(), torch.randn(3).tolist())
    assert first == second


def test_rng_state_survives_a_torch_save_roundtrip(tmp_path):
    seed_all(5)
    path = tmp_path / "rng.pt"
    torch.save({"rng": get_rng_state()}, path)
    expected = torch.randn(3).tolist()
    loaded = torch.load(path, weights_only=False)
    set_rng_state(loaded["rng"])
    assert torch.randn(3).tolist() == expected


# =========================================================================== #
# 3. IO and run identity
# =========================================================================== #
def test_build_run_id_matches_the_frozen_pattern():
    assert build_run_id("gnn", "full", 0) == "gnn_full_s0"
    assert build_run_id("sdf_fno", "aoa", 3, "ablation") == "sdf_fno_aoa_s3_ablation"
    assert build_run_id("gnn", "full", 0, None) == "gnn_full_s0"
    assert build_run_id("gnn", "shape 5", 1, "a/b") == "gnn_shape-5_s1_a-b"
    with pytest.raises(ValueError):
        build_run_id("", "full", 0)


def test_write_json_is_atomic_and_strict(tmp_path):
    path = tmp_path / "deep" / "metrics.json"
    write_json(path, {"a": np.float32(1.5), "b": torch.tensor(2.0),
                      "c": float("nan"), "d": np.arange(3)})
    assert not list(path.parent.glob("*.tmp"))
    obj = read_json(path)
    assert obj["a"] == pytest.approx(1.5)
    assert obj["b"] == pytest.approx(2.0)
    assert obj["c"] is None            # NaN is not valid JSON -> null
    assert obj["d"] == [0, 1, 2]
    json.loads(path.read_text(encoding="utf-8"))  # strict parse must succeed


def test_append_csv_row_writes_one_header(tmp_path):
    path = tmp_path / "history.csv"
    fields = ("epoch", "train_loss", "val_loss")
    append_csv_row(path, {"epoch": 0, "train_loss": 1.0, "val_loss": None}, fields)
    append_csv_row(path, {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4}, fields)
    rows = read_csv(path)
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert rows[0]["val_loss"] == ""
    assert path.read_text(encoding="utf-8").count("epoch") == 1


# =========================================================================== #
# 4. Metrics
# =========================================================================== #
def test_rel_l2_and_mae_are_what_they_claim():
    true = torch.tensor([3.0, 4.0])
    pred = torch.tensor([3.0, 0.0])
    assert rel_l2(pred, true) == pytest.approx(4.0 / 5.0, rel=1e-6)
    assert mae(pred, true) == pytest.approx(2.0)
    assert rel_l2(true, true) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        rel_l2(torch.zeros(3), torch.zeros(4))


def test_spearman_is_rank_based():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x**3) == pytest.approx(1.0)     # monotone, not linear
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(spearman(x, np.ones_like(x)))    # undefined, not 0.0
    assert math.isnan(spearman([1.0], [1.0]))


def test_fsc_and_coef_block():
    cl_int = np.array([1.0, 2.0])
    cl_head = np.array([1.1, 1.7])
    assert fsc(cl_int, cl_head) == pytest.approx([0.1, 0.3])

    block = coef_metrics(
        cl_head=[1.1, 1.7, 3.2], cd_head=[0.02, 0.03, 0.04],
        cl_int=[1.0, 2.0, 3.0], cd_int=[0.021, 0.031, 0.041],
        cl_true=[1.0, 2.0, 3.0], cd_true=[0.02, 0.03, 0.04],
    )
    assert block["cl_int_mae"] == pytest.approx(0.0, abs=1e-12)
    assert block["cd_int_mae"] == pytest.approx(0.001, rel=1e-6)
    assert block["cd_spearman"] == pytest.approx(1.0)
    # aggregate relative error: sum|err| / sum|true|
    assert block["cd_rel"] == pytest.approx(0.003 / 0.09, rel=1e-6)


def test_accumulator_averages_per_sim_not_per_point():
    acc = MetricAccumulator()
    acc.add(p_pred=torch.zeros(1000), p_true=torch.ones(1000),
            cl_head=0.0, cd_head=0.0, cl_int=0.0, cd_int=0.0,
            cl_true=0.0, cd_true=0.0)
    acc.add(p_pred=torch.ones(2), p_true=torch.ones(2),
            cl_head=0.0, cd_head=0.0, cl_int=0.0, cd_int=0.0,
            cl_true=0.0, cd_true=0.0)
    # 1.0 and 0.0 -> 0.5 regardless of the 500x point-count imbalance
    assert acc.field_block()["p_rel_l2"] == pytest.approx(0.5, rel=1e-6)
    assert len(acc.per_sim_rows()) == 2


# =========================================================================== #
# 5. Schema validation
# =========================================================================== #
def test_empty_metrics_is_schema_valid():
    metrics = empty_metrics(run_id="x", model="gnn", split="full")
    assert check_metrics(metrics) == []
    validate_metrics(metrics)          # must not raise


def test_schema_catches_a_missing_key():
    metrics = empty_metrics(run_id="x", model="gnn", split="full")
    del metrics["coef"]["cd_spearman"]
    errors = check_metrics(metrics)
    assert any("coef.cd_spearman" in e for e in errors)
    with pytest.raises(SchemaError, match="cd_spearman"):
        assert_valid_metrics(metrics)


def test_validate_raises_rather_than_returning_errors():
    """A validator whose only signal is a return value gets ignored (A5)."""
    bad_metrics = empty_metrics(run_id="x", model="gnn", split="full")
    del bad_metrics["cost"]
    with pytest.raises(SchemaError, match="cost"):
        validate_metrics(bad_metrics)
    assert validate_metrics is assert_valid_metrics
    # on success the empty error list comes back, so `== []` callers still work
    assert validate_metrics(empty_metrics(run_id="x", model="g", split="f")) == []

    with pytest.raises(SchemaError, match="levels"):
        validate_uq({"ensemble_id": "e", "model": "gnn", "split": "full",
                     "seeds": [0], "method": "conformal",
                     "coef": {}, "field": {}})
    assert check_uq({"ensemble_id": "e"}), "check_* must report, not raise"


def test_schema_catches_a_missing_top_level_block_and_bad_types():
    metrics = empty_metrics(run_id="x", model="gnn", split="full")
    del metrics["consistency"]
    metrics["seed"] = "zero"
    metrics["field"]["p_rel_l2"] = "small"
    errors = check_metrics(metrics)
    assert any("missing key: consistency" in e for e in errors)
    assert any(e.startswith("seed:") for e in errors)
    assert any("field.p_rel_l2" in e for e in errors)


def test_schema_tolerates_extra_keys_unless_strict():
    metrics = empty_metrics(run_id="x", model="gnn", split="full")
    metrics["n_sims"] = 12
    assert check_metrics(metrics) == []
    assert any("n_sims" in e for e in check_metrics(metrics, allow_extra=False))


# =========================================================================== #
# 6. Harness
# =========================================================================== #
def test_evaluate_produces_a_schema_valid_metrics_dict(tmp_path):
    seed_all(0)
    model = TinyModel()
    ds = TinyDataset(n=6)
    metrics = evaluate(
        model, ds, "full",
        run_meta={"run_id": "tiny_full_s0", "model": "tiny", "seed": 0,
                  "epochs": 2, "train_time_s": 1.0},
        device="cpu", batch_size=2, collate_fn=tiny_collate,
        denormalize=False, integrate_fn=tiny_integrate,
        save_dir=tmp_path / "run",
    )
    assert_valid_metrics(metrics)
    assert metrics["run_id"] == "tiny_full_s0"
    assert metrics["split"] == "full"
    assert metrics["n_sims"] == 6
    assert metrics["params"] == sum(p.numel() for p in model.parameters())
    for key in ("p_rel_l2", "tau_rel_l2", "p_mae", "tau_mae"):
        assert metrics["field"][key] is not None and metrics["field"][key] >= 0
    for key in ("cl_int_mae", "cd_int_mae", "cd_spearman"):
        assert metrics["coef"][key] is not None
    assert metrics["consistency"]["fsc_cd"] is not None
    assert metrics["consistency"]["sym_residual"] is not None
    assert metrics["cost"]["infer_ms_per_sim"] > 0
    assert metrics["cost"]["peak_mem_mb"] is None       # CPU run

    saved = read_json(tmp_path / "run" / "metrics.json")
    assert saved["run_id"] == "tiny_full_s0"
    assert (tmp_path / "run" / "per_sim.csv").is_file()
    assert len(read_csv(tmp_path / "run" / "per_sim.csv")) == 6


def test_evaluate_without_physics_leaves_integrated_coefs_null():
    seed_all(0)
    metrics = evaluate(
        TinyModel(), TinyDataset(n=4), "full",
        device="cpu", batch_size=2, collate_fn=tiny_collate,
        denormalize=False, integrate_fn=False, compute_symmetry=False,
    )
    assert_valid_metrics(metrics)
    assert metrics["coef"]["cd_int_mae"] is None
    assert metrics["coef"]["cd_head_mae"] is not None   # head still evaluated
    assert metrics["consistency"]["sym_residual"] is None


def test_reflect_batch_mirrors_geometry_and_condition():
    batch = tiny_collate([make_item(0), make_item(1)])
    r = reflect_batch(batch)
    assert torch.allclose(r["surf_pos"][:, 0], batch["surf_pos"][:, 0])
    assert torch.allclose(r["surf_pos"][:, 1], -batch["surf_pos"][:, 1])
    assert torch.allclose(r["surf_normal"][:, 1], -batch["surf_normal"][:, 1])
    assert torch.allclose(r["cond"][:, 1], -batch["cond"][:, 1])
    assert torch.allclose(r["surf_ds"], batch["surf_ds"])   # scalars untouched
    assert torch.allclose(batch["surf_pos"][:, 1], -r["surf_pos"][:, 1])  # no aliasing


def test_symmetry_residual_is_zero_for_an_equivariant_model():
    """A model whose output is a pointwise function of |y| is exactly equivariant."""

    class EquivariantModel(nn.Module):
        def forward(self, batch):
            pos = batch["surf_pos"]
            n = int(batch["num_graphs"])
            p = pos[:, 0] * 2.0 + pos[:, 1].abs()
            tau = torch.stack([pos[:, 0], pos[:, 1]], dim=1) * 0.1
            return {"p": p, "tau": tau,
                    "coef_head": torch.zeros(n, 2, dtype=pos.dtype)}

    metrics = evaluate(
        EquivariantModel(), TinyDataset(n=3), "full",
        device="cpu", batch_size=3, collate_fn=tiny_collate,
        denormalize=False, integrate_fn=tiny_integrate,
    )
    assert metrics["consistency"]["sym_residual"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["consistency"]["antisym_cl_gap"] == pytest.approx(0.0, abs=1e-6)


# =========================================================================== #
# 7. Baselines
# =========================================================================== #
def test_parse_naca_params_reads_the_trailing_shape_numbers():
    params, flag = baselines_mod.parse_naca_params(
        "airFoil2D_SST_57.872_6.917_3.16_3.542_11.436"
    )
    assert params[:3] == pytest.approx([3.16, 3.542, 11.436])
    assert flag == 0                       # 3 shape numbers -> 4-digit family
    params5, flag5 = baselines_mod.parse_naca_params(
        "airFoil2D_SST_50.0_2.0_1.0_2.0_3.0_12.0"
    )
    assert flag5 == 1
    assert baselines_mod.parse_naca_params("garbage")[0].tolist() == [0.0] * 4


@pytest.mark.parametrize("which", ["constant", "ridge"])
def test_baselines_emit_the_same_metrics_schema(which, tmp_path):
    ds = TinyDataset(n=8)
    metrics = baselines_mod.run_baseline(
        which, ds, ds, "full", seed=0,
        batch_size=4, collate_fn=tiny_collate,
        integrate_fn=tiny_integrate, save_dir=tmp_path / which,
    )
    assert_valid_metrics(metrics)
    assert metrics["model"] == which
    assert metrics["run_id"] == f"{which}_full_s0"
    assert metrics["n_sims"] == 8
    assert metrics["field"]["p_rel_l2"] is not None
    assert (tmp_path / which / "metrics.json").is_file()


def test_ridge_beats_the_constant_predictor_on_a_learnable_signal():
    """C_L is linear in the sim index here, which ridge sees through the name."""
    ds = TinyDataset(n=12)
    const = baselines_mod.fit_constant(ds)
    ridge = baselines_mod.fit_ridge(ds)
    names = [it["sim_name"] for it in ds.items]
    conds = np.stack([it["cond"].numpy() for it in ds.items])
    truth = np.array([[float(it["cl_true"]), float(it["cd_true"])] for it in ds.items])
    err_ridge = np.abs(ridge.predict_coef(names, conds) - truth).mean()
    err_const = np.abs(np.tile(const.coef_mean, (len(ds), 1)) - truth).mean()
    assert err_ridge < err_const


# =========================================================================== #
# 8. Training loop, checkpointing and exact resume
# =========================================================================== #
def _train(tmp_dir, *, epochs, stop_after=None, resume=True, seed=0):
    seed_all(seed)
    model = TinyModel()
    cfg = tiny_cfg(**{"train.epochs": epochs, "seed": seed})
    result = train_mod.train_model(
        model, TinyDataset(n=6), None, cfg,
        ckpt_dir=tmp_dir,
        history_path=Path(tmp_dir) / "history.csv",
        device="cpu",
        collate=tiny_collate,
        denorm=None,
        resume=resume,
        stop_after_epochs=stop_after,
        log=lambda _m: None,
    )
    return model, result


def test_train_model_writes_checkpoints_and_history(tmp_path):
    model, result = _train(tmp_path / "run", epochs=2)
    assert (tmp_path / "run" / "last.pt").is_file()
    assert (tmp_path / "run" / "best.pt").is_file()
    rows = read_csv(tmp_path / "run" / "history.csv")
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert float(rows[0]["train_loss"]) > 0
    assert float(rows[1]["lr"]) < float(rows[0]["lr"])   # cosine decays

    ckpt = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "epoch", "rng", "config", "norm_ref"):
        assert key in ckpt, f"checkpoint is missing {key!r} (CONTEXT.md 10)"
    assert ckpt["epoch"] == 2
    assert result["epochs"] == 2 and result["best_metric"] is not None


def test_checkpoint_resume_is_bit_identical(tmp_path):
    """2 epochs straight == 1 epoch + resume + 1 epoch, parameter for parameter."""
    straight, _ = _train(tmp_path / "straight", epochs=2)

    # Session 1: same 2-epoch config, interrupted after the first epoch.
    partial, _ = _train(tmp_path / "resumed", epochs=2, stop_after=1)
    assert torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)["epoch"] == 1

    # Session 2: fresh process state, auto-resume from last.pt.
    resumed, result = _train(tmp_path / "resumed", epochs=2)
    assert result["epochs_run"] == 1     # only the remaining epoch ran

    a = dict(straight.state_dict())
    b = dict(resumed.state_dict())
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), f"parameter {key} diverged after resume"

    # And the interrupted model really was mid-training, not already converged.
    assert any(not torch.equal(partial.state_dict()[k], a[k]) for k in a)


def test_resume_can_be_disabled(tmp_path):
    _train(tmp_path / "run", epochs=1)
    _, result = _train(tmp_path / "run", epochs=1, resume=False)
    assert result["epochs_run"] == 1     # started from scratch, ran epoch 0 again


def test_finished_run_resumes_to_a_no_op(tmp_path):
    _train(tmp_path / "run", epochs=2)
    _, result = _train(tmp_path / "run", epochs=2)
    assert result["epochs_run"] == 0
    assert result["history"] == []


def test_fallback_loss_is_field_plus_head():
    seed_all(0)
    model = TinyModel()
    batch = tiny_collate([make_item(0), make_item(1)])
    pred = model(batch)
    loss, parts = train_mod.fallback_total_loss(
        pred, batch, {"head": 0.0, "force": 0.0, "sym": 0.0}, {}
    )
    assert loss.requires_grad
    assert parts["loss"] == pytest.approx(parts["p_rel_l2"] + parts["tau_rel_l2"], rel=1e-5)
    assert math.isnan(parts["force"]) and math.isnan(parts["sym"])


def test_head_term_is_on_by_default_and_reaches_the_head():
    """D-016: without lambda_H the coefficient head gets no gradient at all."""
    seed_all(0)
    model = TinyModel()
    batch = tiny_collate([make_item(0), make_item(1)])

    # default weights -> head term present, worth exactly lambda_H after scaling
    loss, parts = train_mod.fallback_total_loss(pred := model(batch), batch, {}, {})
    assert not math.isnan(parts["head"])
    assert parts["loss"] == pytest.approx(
        parts["p_rel_l2"] + parts["tau_rel_l2"] + train_mod.DEFAULT_LAMBDA_H, rel=1e-5
    )
    loss.backward()
    grad = model.coef_head_mlp.weight.grad
    assert grad is not None and grad.abs().sum() > 0

    # and with lambda_H = 0 the head is orphaned -- the failure mode D-016 fixes
    model.zero_grad(set_to_none=True)
    loss0, _ = train_mod.fallback_total_loss(model(batch), batch, {"head": 0.0}, {})
    loss0.backward()
    orphan = model.coef_head_mlp.weight.grad
    assert orphan is None or orphan.abs().sum() == 0


def test_head_regression_loss_is_zero_on_perfect_coefficients():
    batch = tiny_collate([make_item(0), make_item(1)])
    perfect = {
        "coef_head": torch.stack([batch["cl_true"], batch["cd_true"]], dim=-1),
    }
    assert float(train_mod.head_regression_loss(perfect, batch)) == pytest.approx(0.0)


def test_fallback_loss_symmetry_term_is_normalized_at_init():
    seed_all(0)
    model = TinyModel()
    batch = tiny_collate([make_item(0), make_item(1)])
    pred = model(batch)
    norm_ref: dict[str, float] = {}
    data_only, _ = train_mod.fallback_total_loss(pred, batch, {}, {})
    with_sym, parts = train_mod.fallback_total_loss(
        pred, batch, {"sym": 1.0}, norm_ref, model=model
    )
    assert "sym" in norm_ref                       # captured on the first batch
    # term / its own init value == 1, so the auxiliary starts at order one
    assert float(with_sym.detach()) == pytest.approx(float(data_only.detach()) + 1.0,
                                                     rel=1e-5)


def test_build_optimizer_and_scheduler_follow_the_frozen_recipe():
    model = TinyModel()
    cfg = tiny_cfg(**{"train.lr": 3e-4, "train.epochs": 10})
    opt = train_mod.build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    sched = train_mod.build_scheduler(opt, cfg, 10)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 10


def test_build_model_reports_unknown_names_clearly():
    with pytest.raises(train_mod.ModuleNotReadyError, match="totally_not_a_model"):
        train_mod.build_model({"model": {"name": "totally_not_a_model"}})


def test_parse_cli_accepts_overrides_anywhere_on_the_line():
    """argparse fills nargs='*' greedily; overrides must survive interleaving."""
    parser = train_mod.build_parser()
    args = train_mod.parse_cli(parser, [
        "--config", "configs/gnn.yaml", "split=aoa", "--device", "cpu",
        "seed=3", "--dry-run", "train.lr=3e-4",
    ])
    assert args.config == "configs/gnn.yaml"
    assert args.device == "cpu" and args.dry_run is True
    assert set(args.overrides) == {"split=aoa", "seed=3", "train.lr=3e-4"}


def test_parse_cli_still_rejects_real_typos(capsys):
    parser = train_mod.build_parser()
    with pytest.raises(SystemExit):
        train_mod.parse_cli(parser, ["--config", "configs/gnn.yaml", "--not-a-flag"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_batch_keys_are_passed_through_untouched(tmp_path):
    """D-017: precomputed grid_sdf / edge_index / curvature must reach the model."""
    seen: dict = {}

    class KeySpy(nn.Module):
        def forward(self, batch):
            seen.update({k: True for k in batch})
            n = int(batch["num_graphs"])
            pos = batch["surf_pos"]
            return {"p": pos[:, 0].clone(), "tau": pos.clone(),
                    "coef_head": torch.zeros(n, 2, dtype=pos.dtype)}

    def collate_with_extras(items):
        batch = tiny_collate(items)
        n_pts = batch["surf_pos"].shape[0]
        batch["curvature"] = torch.zeros(n_pts)
        batch["grid_sdf"] = torch.zeros(len(items), 8, 8)
        batch["edge_index"] = torch.zeros(2, 4, dtype=torch.long)
        return batch

    evaluate(KeySpy(), TinyDataset(n=2), "full", device="cpu", batch_size=2,
             collate_fn=collate_with_extras, denormalize=False,
             integrate_fn=tiny_integrate, compute_symmetry=False)
    for key in ("curvature", "grid_sdf", "edge_index"):
        assert seen.get(key), f"{key} was stripped before reaching the model"


# =========================================================================== #
# 9. Sweep planning
# =========================================================================== #
def test_expand_spec_handles_matrix_and_runs():
    spec = {
        "name": "t",
        "defaults": {"overrides": ["train.epochs=5"]},
        "matrix": {"config": ["configs/gnn.yaml"],
                   "overrides": {"split": ["full", "aoa"], "seed": [0, 1]}},
        "runs": [{"config": "configs/transolver.yaml", "overrides": ["split=full"]}],
    }
    runs = sweep_mod.expand_spec(spec)
    assert len(runs) == 5
    assert all(r["overrides"][0] == "train.epochs=5" for r in runs)
    assert runs[0]["overrides"][1:] == ["split=full", "seed=0"]
    assert runs[-1]["config"] == "configs/transolver.yaml"


def test_core_sweep_spec_covers_the_planned_grid():
    spec = sweep_mod.load_spec(ROOT / "configs" / "sweeps" / "core.yaml")
    runs = sweep_mod.expand_spec(spec)
    assert len(runs) == 18                       # 3 models x 6 splits x 1 seed
    ids = {sweep_mod.run_id_for(r["config"], r["overrides"]) for r in runs}
    assert "gnn_full_s0" in ids and "sdf_fno_combined_s0" in ids
    assert len(ids) == 18


def test_sweep_skips_runs_that_already_have_metrics(tmp_path, capsys):
    results = tmp_path / "results"
    (results / "gnn_full_s0").mkdir(parents=True)
    write_json(results / "gnn_full_s0" / "metrics.json",
               empty_metrics(run_id="gnn_full_s0", model="gnn", split="full"))
    spec = tmp_path / "s.yaml"
    spec.write_text(
        "name: t\nruns:\n"
        "  - config: configs/gnn.yaml\n    overrides: [split=full]\n"
        "  - config: configs/gnn.yaml\n    overrides: [split=aoa]\n",
        encoding="utf-8",
    )
    rc = sweep_mod.main(["--spec", str(spec), "--dry-run",
                         "--results-root", str(results)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 already complete, 1 to run" in out
    assert "[done] gnn_full_s0" in out
    assert "[todo] gnn_aoa_s0" in out


def test_sweep_forwards_train_args_as_one_shlex_string(tmp_path, capsys):
    spec = tmp_path / "s.yaml"
    spec.write_text(
        "name: t\nruns:\n  - config: configs/gnn.yaml\n    overrides: [split=full]\n",
        encoding="utf-8",
    )
    rc = sweep_mod.main(["--spec", str(spec), "--dry-run",
                         "--results-root", str(tmp_path / "r"),
                         "--train-args=--device cpu --no-eval"])
    assert rc == 0


def test_sweep_stops_cleanly_when_the_wall_clock_budget_is_gone(tmp_path, capsys):
    """--max-seconds must stop before starting a run it cannot finish."""
    spec = tmp_path / "s.yaml"
    spec.write_text(
        "name: t\nruns:\n  - config: configs/gnn.yaml\n    overrides: [split=full]\n",
        encoding="utf-8",
    )
    rc = sweep_mod.main([
        "--spec", str(spec), "--results-root", str(tmp_path / "results"),
        "--max-seconds", "10", "--reserve-seconds", "10",  # zero budget left
    ])
    out = capsys.readouterr().out
    assert rc == 0                                   # clean stop, not a failure
    assert "budget exhausted" in out
    assert "1 run(s) not started" in out
    assert "0 ok, 0 failed, 1 not started" in out
