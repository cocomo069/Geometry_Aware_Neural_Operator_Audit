"""CPU tests for the three surrogate architectures (PLAN Phase-1 row A3).

Everything here runs on CPU with synthetic NACA-like airfoils, per CONTEXT.md
section 11.  Model-capacity tests use the *default* configs (that is the point
of the budget assertion); everything else uses deliberately tiny widths so the
suite stays fast.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models import MODEL_REGISTRY, available_models, build_model  # noqa: E402
from src.models.common import (  # noqa: E402
    FourierFeatures,
    build_mlp,
    cond_features,
    count_params,
    knn_edges,
    segment_max,
    segment_mean,
    segment_sum,
    to_dense_batch,
)
from src.models.heads import CoefHead  # noqa: E402
from src.models.losses import (  # noqa: E402
    init_norm_ref,
    rel_l2_per_sample,
    total_loss,
)
from src.models.sdf_fno import (  # noqa: E402
    CONDITIONINGS,
    SpectralConv2d,
    compute_grid_sdf,
    make_latent_grid,
)

PARAM_TARGET = 1_500_000
PARAM_TOLERANCE = 0.20

#: tiny configs -- same architecture, ~100x fewer parameters, CPU-fast
TINY_CONFIGS = {
    "gnn": {"hidden_dim": 16, "mlp_hidden": 16, "n_rounds": 2, "k": 6,
            "n_freqs": 2, "trunk_dim": 16, "coef_hidden": 16, "coef_layers": 1},
    "sdf_fno": {"width": 8, "modes": 4, "n_fno_layers": 2, "spectral_groups": 2,
                "grid_res": (16, 16), "k_enc": 4, "k_dec": 4, "kernel_hidden": 16,
                "point_hidden": 16, "n_freqs": 2, "trunk_dim": 16,
                "coef_hidden": 16, "coef_layers": 1},
    "transolver": {"dim": 16, "n_layers": 2, "n_heads": 2, "n_slices": 4,
                   "ffn_mult": 2, "n_freqs": 2, "trunk_dim": 16,
                   "coef_hidden": 16, "coef_layers": 1},
}


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #

def naca4(m: float, p: float, t: float, n: int) -> np.ndarray:
    """Ordered closed NACA-4-digit contour, ``(n, 2)``, unit chord, TE -> TE."""
    half = n // 2 + 1
    beta = np.linspace(0.0, np.pi, half)
    x = (1.0 - np.cos(beta)) / 2.0
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    pc = max(p, 1e-6)
    yc = np.where(x < pc,
                  m / pc ** 2 * (2 * pc * x - x ** 2),
                  m / (1 - pc) ** 2 * ((1 - 2 * pc) + 2 * pc * x - x ** 2))
    dyc = np.where(x < pc,
                   2 * m / pc ** 2 * (pc - x),
                   2 * m / (1 - pc) ** 2 * (pc - x))
    th = np.arctan(dyc)
    upper = np.stack([x - yt * np.sin(th), yc + yt * np.cos(th)], axis=-1)[::-1]
    lower = np.stack([x + yt * np.sin(th), yc - yt * np.cos(th)], axis=-1)[1:]
    contour = np.concatenate([upper, lower], axis=0)
    return np.ascontiguousarray(contour[:n], dtype=np.float64)


def make_sample(n_points: int, seed: int = 0, cond=(50.0, 3.0)) -> dict:
    """One synthetic airfoil sample with all the frozen surface keys."""
    rng = np.random.default_rng(seed)
    shape = 0.01 + 0.03 * rng.random(), 0.3 + 0.2 * rng.random(), 0.09 + 0.06 * rng.random()
    pos = naca4(*shape, n=n_points)
    tangent = np.roll(pos, -1, axis=0) - np.roll(pos, 1, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    tangent = tangent / length
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)
    ds = length.squeeze(-1) / 2.0
    to32 = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return {
        "surf_pos": to32(pos),
        "surf_normal": to32(normal),
        "surf_ds": to32(ds),
        "curvature": to32(rng.normal(size=n_points) * 0.05),
        "surf_p": to32(rng.normal(size=n_points)),
        "surf_tau": to32(rng.normal(size=(n_points, 2))),
        "cond": to32(np.asarray(cond)).view(1, 2),
        "cl_true": torch.tensor([0.4], dtype=torch.float32),
        "cd_true": torch.tensor([0.02], dtype=torch.float32),
    }


def collate(samples) -> dict:
    """Concatenating collate mirroring ``src/data/airfrans_loader.py`` (CONTEXT s.4)."""
    out: dict = {}
    keys = ("surf_pos", "surf_normal", "surf_ds", "curvature", "surf_p", "surf_tau",
            "cond", "cl_true", "cd_true")
    for key in keys:
        out[key] = torch.cat([s[key] for s in samples], dim=0)
    out["batch_idx"] = torch.cat(
        [torch.full((s["surf_pos"].shape[0],), i, dtype=torch.long)
         for i, s in enumerate(samples)], dim=0)
    return out


@pytest.fixture(scope="module")
def batch() -> dict:
    """Two airfoils, 100 + 120 surface points."""
    return collate([make_sample(100, seed=0, cond=(50.0, 3.0)),
                    make_sample(120, seed=1, cond=(40.0, -2.0))])


@pytest.fixture(params=sorted(TINY_CONFIGS))
def model_name(request) -> str:
    return request.param


def tiny(name: str, **overrides):
    cfg = dict(TINY_CONFIGS[name])
    cfg.update(overrides)
    torch.manual_seed(0)
    return build_model(name, cfg)


# --------------------------------------------------------------------------- #
# common.py building blocks
# --------------------------------------------------------------------------- #

def test_fourier_features_shape_and_determinism():
    ff = FourierFeatures(2, n_freqs=4)
    assert ff.out_dim == 2 * (1 + 2 * 4)
    x = torch.randn(7, 2)
    assert ff(x).shape == (7, ff.out_dim)
    assert torch.equal(ff(x), ff(x))
    with pytest.raises(ValueError):
        ff(torch.randn(7, 3))


def test_build_mlp_shapes():
    mlp = build_mlp(5, 3, 8, n_hidden_layers=2)
    assert mlp(torch.randn(4, 5)).shape == (4, 3)
    assert count_params(mlp) == (5 * 8 + 8) + (8 * 8 + 8) + (8 * 3 + 3)
    with pytest.raises(ValueError):
        build_mlp(0, 3, 8)


def test_segment_reductions_mask_empty_segments():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 2])
    assert torch.allclose(segment_sum(src, idx, 4).squeeze(-1),
                          torch.tensor([4.0, 0.0, 5.0, 0.0]))
    assert torch.allclose(segment_mean(src, idx, 4).squeeze(-1),
                          torch.tensor([2.0, 0.0, 5.0, 0.0]))
    assert torch.allclose(segment_max(src, idx, 4).squeeze(-1),
                          torch.tensor([3.0, 0.0, 5.0, 0.0]))
    assert torch.isfinite(segment_mean(src, idx, 4)).all()


def test_to_dense_batch_roundtrip():
    x = torch.randn(7, 3)
    bidx = torch.tensor([0, 0, 0, 1, 2, 2, 2])
    dense, mask, slot = to_dense_batch(x, bidx, 3)
    assert dense.shape == (3, 3, 3) and mask.shape == (3, 3)
    assert mask.sum().item() == 7
    assert torch.equal(dense[bidx, slot], x)
    assert torch.equal(dense[1, 1:], torch.zeros(2, 3))   # padding stays zero


def test_cond_features():
    cond = torch.tensor([[3.0, 4.0]])
    feats = cond_features(cond)
    assert feats.shape == (1, 5)
    assert pytest.approx(feats[0, 2].item(), abs=1e-6) == 5.0
    assert pytest.approx(feats[0, 3].item(), abs=1e-6) == 0.6


def test_knn_edges_respects_graph_boundaries():
    pos = torch.randn(30, 2)
    bidx = torch.cat([torch.zeros(10, dtype=torch.long),
                      torch.ones(20, dtype=torch.long)])
    edge = knn_edges(pos, k=4, batch_idx=bidx, num_graphs=2)
    assert edge.shape == (2, 30 * 4)
    assert torch.equal(bidx[edge[0]], bidx[edge[1]])       # no cross-graph edges
    assert (edge[0] != edge[1]).all()                      # no self loops


# --------------------------------------------------------------------------- #
# forward / backward on every model
# --------------------------------------------------------------------------- #

def test_forward_shapes(model_name, batch):
    model = tiny(model_name)
    out = model(batch)
    n = batch["surf_pos"].shape[0]
    b = batch["cond"].shape[0]
    assert set(out) == {"p", "tau", "coef_head"}
    assert out["p"].shape == (n,)
    assert out["tau"].shape == (n, 2)
    assert out["coef_head"].shape == (b, 2)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_backward_gives_finite_grads_for_all_params(model_name, batch):
    model = tiny(model_name)
    out = model(batch)
    loss = out["p"].pow(2).mean() + out["tau"].pow(2).mean() + out["coef_head"].pow(2).mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f"parameters with no grad: {missing}"
    assert not nonfinite, f"parameters with non-finite grad: {nonfinite}"


def test_deterministic_under_seed(model_name, batch):
    a = tiny(model_name)(batch)
    b = tiny(model_name)(batch)
    for key in ("p", "tau", "coef_head"):
        assert torch.equal(a[key], b[key]), f"{model_name}/{key} not reproducible"


def test_curvature_optional(model_name, batch):
    """A loader that does not yet ship curvature must still work (zeros fallback)."""
    stripped = {k: v for k, v in batch.items() if k != "curvature"}
    out = tiny(model_name)(stripped)
    assert torch.isfinite(out["p"]).all()


def test_batch_validation_rejects_bad_shapes(model_name, batch):
    model = tiny(model_name)
    bad = dict(batch)
    bad["surf_normal"] = batch["surf_normal"][:-1]
    with pytest.raises(ValueError):
        model(bad)
    missing = {k: v for k, v in batch.items() if k != "cond"}
    with pytest.raises(KeyError):
        model(missing)


def test_eval_mode_matches_train_mode(model_name, batch):
    """No dropout / batchnorm anywhere, so the two modes must agree exactly."""
    model = tiny(model_name)
    model.train()
    train_out = model(batch)["p"]
    model.eval()
    with torch.no_grad():
        eval_out = model(batch)["p"]
    assert torch.allclose(train_out, eval_out, atol=1e-6)


# --------------------------------------------------------------------------- #
# batched-vs-single consistency (M3 slice pooling is the interesting case)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(TINY_CONFIGS))
def test_batched_matches_single_sample(name, batch):
    """Padding + masking must not leak between samples.

    For M3 this specifically exercises the physics-attention slice pooling:
    ``z_m`` is a mask-weighted mean over the points of *one* graph, so a sample
    run alone has to reproduce its rows from the batched run.
    """
    model = tiny(name)
    model.eval()
    with torch.no_grad():
        batched = model(batch)
    counts = torch.bincount(batch["batch_idx"], minlength=2).tolist()
    start = 0
    for g, n in enumerate(counts):
        single = {
            "surf_pos": batch["surf_pos"][start:start + n],
            "surf_normal": batch["surf_normal"][start:start + n],
            "surf_ds": batch["surf_ds"][start:start + n],
            "curvature": batch["curvature"][start:start + n],
            "cond": batch["cond"][g:g + 1],
            "batch_idx": torch.zeros(n, dtype=torch.long),
        }
        with torch.no_grad():
            alone = model(single)
        assert torch.allclose(alone["p"], batched["p"][start:start + n], atol=2e-5), name
        assert torch.allclose(alone["tau"], batched["tau"][start:start + n], atol=2e-5)
        assert torch.allclose(alone["coef_head"], batched["coef_head"][g:g + 1], atol=2e-5)
        start += n


def test_physics_attention_never_materializes_n_by_n():
    """M3 memory is O(N*M + M^2): assert the assignment map is (B, H, N, M)."""
    from src.models.geo_transformer import PhysicsAttention

    attn = PhysicsAttention(dim=16, n_heads=2, n_slices=4)
    seen = {}
    original = torch.softmax

    def spy(x, dim=None, **kwargs):
        seen.setdefault("shapes", []).append(tuple(x.shape))
        return original(x, dim=dim, **kwargs)

    torch.softmax = spy
    try:
        out = attn(torch.randn(2, 50, 16), torch.ones(2, 50, dtype=torch.bool))
    finally:
        torch.softmax = original
    assert out.shape == (2, 50, 16)
    assert seen["shapes"] == [(2, 2, 50, 4)]          # (B, H, N, M), never (N, N)


def test_physics_attention_handles_large_point_counts():
    """Large N must stay tractable: the cost is O(N*M), not O(N^2)."""
    from src.models.geo_transformer import PhysicsAttention

    attn = PhysicsAttention(dim=16, n_heads=2, n_slices=4)
    for n in (64, 512):
        out = attn(torch.randn(1, n, 16), torch.ones(1, n, dtype=torch.bool))
        assert out.shape == (1, n, 16)


# --------------------------------------------------------------------------- #
# parameter budgets at DEFAULT configs (CONTEXT.md section 7)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_param_budget_at_default_config(name):
    torch.manual_seed(0)
    n_params = count_params(build_model(name))
    lo = PARAM_TARGET * (1 - PARAM_TOLERANCE)
    hi = PARAM_TARGET * (1 + PARAM_TOLERANCE)
    assert lo <= n_params <= hi, (
        f"{name}: {n_params:,} params outside 1.5M +/-20% "
        f"([{lo:,.0f}, {hi:,.0f}])")


def test_registry_covers_the_three_architectures():
    assert available_models() == ["gnn", "sdf_fno", "transolver"]
    assert build_model("m1").__class__ is MODEL_REGISTRY["gnn"]
    assert build_model("m3").__class__ is MODEL_REGISTRY["transolver"]
    with pytest.raises(KeyError):
        build_model("nope")


def test_all_models_share_the_same_coef_head_class():
    """FSC is only an architecture comparison if the head is literally shared."""
    for name in MODEL_REGISTRY:
        model = tiny(name)
        assert isinstance(model.coef_head, CoefHead)


# --------------------------------------------------------------------------- #
# M2 specifics: SDF, spectral conv, conditioning ablation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("conditioning", list(CONDITIONINGS))
def test_sdf_fno_conditioning_switch(conditioning, batch):
    model = tiny("sdf_fno", conditioning=conditioning)
    out = model(batch)
    assert out["p"].shape == (batch["surf_pos"].shape[0],)
    assert torch.isfinite(out["p"]).all()
    loss = out["p"].pow(2).mean() + out["coef_head"].pow(2).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_sdf_fno_rejects_unknown_conditioning():
    with pytest.raises(KeyError):
        tiny("sdf_fno", conditioning="occupancy_field")


def test_grid_sdf_sign_convention():
    """Negative inside the body, positive in the fluid, ~0 on the surface."""
    poly = naca4(0.0, 0.4, 0.12, n=200)
    grid = make_latent_grid((-0.5, 1.5, -1.0, 1.0), (64, 64))
    sdf = compute_grid_sdf(poly, grid)
    assert sdf.shape == (64 * 64,)
    inside = grid[sdf < 0]
    assert inside.shape[0] > 0
    assert inside[:, 0].min() >= -0.05 and inside[:, 0].max() <= 1.05
    on_surface = compute_grid_sdf(poly, poly[::7])
    assert np.abs(on_surface).max() < 1e-6
    far = compute_grid_sdf(poly, np.array([[-0.5, -1.0]]))
    assert far[0] > 0.5


def test_grid_sdf_matches_local_fallback():
    """A2's sdf.py and the in-module numpy fallback must agree."""
    poly = naca4(0.02, 0.4, 0.12, n=120)
    grid = make_latent_grid((-0.5, 1.5, -1.0, 1.0), (32, 32))
    external = compute_grid_sdf(poly, grid, use_external=True)
    local = compute_grid_sdf(poly, grid, use_external=False)
    assert np.abs(external - local).max() < 1e-6


def test_precomputed_grid_sdf_is_used(batch):
    """The loader may ship ``grid_sdf``; a wrong-sized one must be rejected."""
    model = tiny("sdf_fno")
    n_grid = model.n_grid
    ok = dict(batch)
    ok["grid_sdf"] = torch.zeros(2, n_grid)
    assert torch.isfinite(model(ok)["p"]).all()
    bad = dict(batch)
    bad["grid_sdf"] = torch.zeros(2, n_grid + 3)
    with pytest.raises(ValueError):
        model(bad)


def test_spectral_conv_shapes_and_param_count():
    conv = SpectralConv2d(8, 8, modes1=4, modes2=4, groups=2)
    x = torch.randn(2, 8, 16, 16)
    assert conv(x).shape == (2, 8, 16, 16)
    # 2 k1-blocks * groups * (in/g) * (out/g) * m1 * m2 * 2 (re, im)
    assert count_params(conv) == 2 * 2 * 4 * 4 * 4 * 4 * 2
    with pytest.raises(ValueError):
        SpectralConv2d(8, 8, 4, 4, groups=3)


def test_spectral_conv_is_translation_equivariant():
    """A spectral multiplier commutes with circular shifts -- a real FNO check."""
    torch.manual_seed(0)
    conv = SpectralConv2d(4, 4, modes1=3, modes2=3, groups=1)
    x = torch.randn(1, 4, 16, 16)
    shifted = torch.roll(x, shifts=(3, 5), dims=(2, 3))
    assert torch.allclose(conv(shifted), torch.roll(conv(x), (3, 5), (2, 3)), atol=1e-5)


def test_sdf_fno_handles_more_points_than_grid_nodes(batch):
    """Discretization invariance in the trivial sense: query count is free."""
    model = tiny("sdf_fno", grid_res=(8, 8))
    assert torch.isfinite(model(batch)["p"]).all()


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #

def test_coef_head_pooling_variants(batch):
    for pooling in ("mean", "max", "mean_max", "ds_mean", "ds_mean_max"):
        head = CoefHead(6, hidden_dim=8, n_hidden_layers=1, pooling=pooling)
        feat = torch.randn(batch["surf_pos"].shape[0], 6)
        out = head(feat, batch["batch_idx"], 2, batch["cond"], ds=batch["surf_ds"])
        assert out.shape == (2, 2)


def test_coef_head_is_permutation_invariant(batch):
    head = CoefHead(6, hidden_dim=8, n_hidden_layers=1, pooling="ds_mean_max")
    n = batch["surf_pos"].shape[0]
    feat = torch.randn(n, 6)
    ref = head(feat, batch["batch_idx"], 2, batch["cond"], ds=batch["surf_ds"])
    perm = torch.cat([torch.randperm(100), 100 + torch.randperm(n - 100)])
    shuffled = head(feat[perm], batch["batch_idx"][perm], 2, batch["cond"],
                    ds=batch["surf_ds"][perm])
    assert torch.allclose(ref, shuffled, atol=1e-5)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #

def test_rel_l2_per_sample_is_exact():
    pred = torch.tensor([3.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 1.0])
    bidx = torch.tensor([0, 0, 1])
    out = rel_l2_per_sample(pred, target, bidx, 2)
    assert out.shape == (2,)
    assert out[0].item() > 1e6          # zero-norm target -> eps-guarded, finite
    assert torch.isfinite(out).all()
    assert pytest.approx(out[1].item(), abs=1e-5) == 0.0


def test_total_loss_defaults_to_pure_data_loss(batch):
    model = tiny("gnn")
    pred = model(batch)
    loss, terms = total_loss(pred, batch)
    assert loss.requires_grad and torch.isfinite(loss)
    assert set(terms) >= {"data_p", "data_tau", "data", "loss"}
    assert "force" not in terms and "sym" not in terms   # lambdas default to 0
    loss.backward()
    # the pure data loss touches the field path only; the coefficient head is
    # trained by the force term (or by an explicit coef loss in the trainer)
    field_path = list(model.encoder.parameters()) + list(model.field_head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in field_path)


def test_total_loss_force_term_uses_injected_integrator(batch):
    model = tiny("gnn")
    pred = model(batch)

    calls = []

    def fake_integrate(p, tau, normal, ds, cond, batch_idx):
        """Stand-in for src.physics.force_integration.integrate_forces."""
        calls.append(p.shape)
        force = segment_sum((-p.unsqueeze(-1) * normal + tau) * ds.unsqueeze(-1),
                            batch_idx, cond.shape[0])
        return {"cl": force[:, 1], "cd": force[:, 0]}

    weights = {"force": 1.0}
    refs = init_norm_ref(pred, batch, weights, integrate_fn=fake_integrate)
    assert "force" in refs and refs["force"] > 0
    loss, terms = total_loss(pred, batch, weights, refs, integrate_fn=fake_integrate)
    assert calls, "integrate_fn was never called"
    assert "force_scaled" in terms
    assert pytest.approx(terms["force_scaled"].item(), rel=1e-5) == 1.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_total_loss_requires_hooks_when_lambda_nonzero(batch):
    pred = tiny("gnn")(batch)
    with pytest.raises(ValueError):
        total_loss(pred, batch, {"force": 0.1})
    with pytest.raises(ValueError):
        total_loss(pred, batch, {"sym": 0.1})


def test_total_loss_symmetry_hook(batch):
    pred = tiny("gnn")(batch)

    def sym_fn(prediction, bat):
        return prediction["p"].abs().mean()

    weights = {"sym": 0.5}
    refs = init_norm_ref(pred, batch, weights, symmetry_fn=sym_fn)
    loss, terms = total_loss(pred, batch, weights, refs, symmetry_fn=sym_fn)
    assert pytest.approx(terms["sym_scaled"].item(), rel=1e-5) == 1.0
    assert torch.isfinite(loss)


def test_total_loss_tolerates_missing_targets(batch):
    """Unlabelled pool geometries (active learning) carry no surf_p / surf_tau."""
    pred = tiny("gnn")(batch)
    unlabelled = {k: v for k, v in batch.items()
                  if k not in ("surf_p", "surf_tau", "cl_true", "cd_true")}
    loss, terms = total_loss(pred, unlabelled)
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_gnn_gradient_checkpointing_is_numerically_transparent(batch):
    """``grad_checkpoint`` is a memory/compute trade, not a model change.

    M1 is the memory-hungry model on the 4 GB card (activations scale as
    ``k * sumN * mlp_hidden`` per round); checkpointing must leave both the
    outputs and the gradients bit-comparable.
    """
    plain = tiny("gnn")
    ckpt = tiny("gnn", grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    plain.train()
    ckpt.train()

    outs = []
    for model in (plain, ckpt):
        out = model(batch)
        (out["p"].pow(2).mean() + out["tau"].pow(2).mean()
         + out["coef_head"].pow(2).mean()).backward()
        outs.append(out)
    assert torch.allclose(outs[0]["p"], outs[1]["p"], atol=1e-6)
    for (name, a), (_, b) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name
