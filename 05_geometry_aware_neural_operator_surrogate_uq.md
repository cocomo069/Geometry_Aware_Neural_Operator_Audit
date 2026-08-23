# Project 5: Geometry-Aware Neural Operator Surrogates for External Aerodynamics

**Author:** Taimoor Amin (BSc Mechanical Engineering, NUST; minor in Machine Learning)
**Status:** Project design document. Solo, 8 to 12 weeks.
**Target:** arXiv preprint by late November 2026, then workshop, conference or journal.

---

## 1. Title and abstract

**Working title:** *Geometry-aware neural operator surrogates for external aerodynamics: physics-consistency, out-of-distribution generalization and calibrated uncertainty on AirfRANS and a DrivAerNet++ subset.*

**Alternate:** *Trust but verify: consistency, OOD behaviour and calibration of geometry-conditioned neural surrogates for aerodynamic fields.*

**Abstract (draft; no numbers until runs finish).** Geometry-conditioned neural surrogates now predict surface pressure and drag on airfoils and cars fast enough to sit inside a design loop. What is far less clear is whether the predictions are usable: whether the predicted fields integrate to the forces the same model reports, whether accuracy survives a shift in Reynolds number or shape family, and whether any reported uncertainty means anything. We evaluate three geometry-conditioning families, a point-cloud graph baseline, a signed-distance-field conditioned Fourier neural operator in the GINO style, and a Transolver-style geometry transformer, on AirfRANS (2D incompressible steady RANS over NACA airfoils) and on a compute-feasible subset of DrivAerNet++ (3D car surfaces). Beyond pointwise field error we report checks the machine learning literature usually skips: (i) force self-consistency, the gap between lift and drag obtained by integrating predicted surface fields and the coefficients the model predicts directly; (ii) boundary and symmetry residuals; (iii) out-of-distribution generalization across Reynolds number, angle of attack and shape family; (iv) calibrated uncertainty from deep ensembles plus split conformal prediction, reported as empirical coverage on fields and on integrated coefficients; (v) data-efficiency curves down to a few dozen training cases. Finally we run a small uncertainty-driven active learning loop whose proposed cases are verified with our own Ansys Fluent RANS simulations and released as an independently generated verification set. This is an evaluation and methodology paper: the contribution is a reproducible consistency-and-calibration protocol plus evidence about where current geometry-aware operators break.

---

## 2. Problem statement

The external aerodynamics surrogate literature has converged on a narrow scoreboard: relative L2 error on velocity and pressure fields, plus an error or rank correlation on drag. Papers get accepted on that scoreboard. Engineers cannot use it.

**Gap 1: internal inconsistency is invisible.** Models are trained with a field loss and, sometimes, a separate head for integrated coefficients. Nothing forces the predicted surface pressure and wall shear stress to integrate to the predicted drag. A model can be excellent on field L2 and still produce a badly wrong force integral, because force is a small difference of large, partially cancelling contributions dominated by regions (leading-edge suction peak, separation onset, base pressure) that carry little weight in a domain-averaged norm. That internal gap is a cheap, ground-truth-free diagnostic almost nobody reports.

**Gap 2: OOD failure is acknowledged but not characterized.** The 2025 Communications Engineering benchmark states plainly that out-of-distribution generalization remains a challenge for every method tested. That is a conclusion, not a characterization. Which axis of shift hurts most: Reynolds number, angle of attack, or shape family? Does error grow smoothly, or fail discontinuously once separation appears? Does it concentrate in the field metric or in the force? Does the geometry encoding change the answer?

**Gap 3: uncertainty is absent or uncalibrated.** Ensemble variance is a heuristic score, least trustworthy under exactly the shift where it matters. Split conformal prediction converts any heuristic score into finite-sample marginal coverage under exchangeability: a genuine guarantee in-distribution, and a measurable failure mode out of it. Conformal coverage for aerodynamic surface fields and derived coefficients has not been systematically reported.

This project attacks all three with open data, small compute, and one asset most machine learning authors lack: the ability to generate independent CFD verification cases in Ansys Fluent, so the active learning loop closes against a solver rather than against held-out data from the same generator.

---

## 3. Background and literature

**Operator learning.** Neural operators learn maps between function spaces rather than between fixed-dimensional vectors, giving discretization invariance in principle. Kovachki et al. give the framework and approximation theory ([JMLR 24(89), 2023](https://jmlr.org/papers/v24/21-1524.html), [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)). The Fourier neural operator of Li et al. parameterizes the kernel integral in Fourier space and is the workhorse on regular grids ([ICLR 2021, arXiv:2010.08895](https://arxiv.org/abs/2010.08895)). DeepONet, from Lu, Jin, Pang, Zhang and Karniadakis, splits the map into branch and trunk networks ([Nature Machine Intelligence 3, 218-229, 2021](https://www.nature.com/articles/s42256-021-00302-5), [arXiv:1910.03193](https://arxiv.org/abs/1910.03193)). Physics-informed variants add PDE residuals to the loss, for example PINO ([arXiv:2111.03794](https://arxiv.org/abs/2111.03794)).

**Geometry conditioning.** Plain FNO needs a regular grid, which external aerodynamics does not have. GINO couples a graph neural operator on the near-body point cloud to an FNO on a regular latent grid, conditions on a signed distance function, and was demonstrated on car surface pressure and drag with a large reported speed-up over GPU CFD ([NeurIPS 2023, arXiv:2309.00583](https://arxiv.org/abs/2309.00583), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)). Transolver introduces physics attention, softly assigning mesh points to learnable slices sharing a physical state, giving transformer-quality results on unstructured meshes at manageable cost ([ICML 2024 spotlight, arXiv:2402.02366](https://arxiv.org/abs/2402.02366), [code](https://github.com/thuml/Transolver), [PMLR v235](https://proceedings.mlr.press/v235/wu24r.html)). Mesh-based message passing in the MeshGraphNets line remains a strong baseline ([arXiv:2010.03409](https://arxiv.org/abs/2010.03409)), as do point-cloud encoders such as PointNet++ ([arXiv:1706.02413](https://arxiv.org/abs/1706.02413)).

**Datasets.** AirfRANS provides 1000 incompressible steady RANS solutions over NACA 4 and 5 digit airfoils, Reynolds roughly 2 to 6 million, angle of attack roughly -5 to 15 degrees, around 32000 mesh points per case, with velocity, pressure and turbulent-viscosity fields, surface quantities, and four predefined ML tasks named `full`, `scarce`, `reynolds` and `aoa`, the last two explicit extrapolation splits ([NeurIPS 2022 Datasets and Benchmarks](https://papers.nips.cc/paper_files/paper/2022/hash/94ab7b23a345f93333eac8748a66c763-Abstract-Datasets_and_Benchmarks.html), [arXiv:2212.07564](https://arxiv.org/abs/2212.07564), [library](https://github.com/Extrality/airfrans_lib), [docs](https://airfrans.readthedocs.io/en/latest/notes/introduction.html), licence ODbL-1.0). DrivAerNet++ provides roughly 8000 car designs with high-fidelity CFD, surface pressure and wall shear stress, point clouds, parametric models, part segmentation and drag and lift coefficients, across fastback, notchback and estateback families with varying underbody and wheel designs ([NeurIPS 2024 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/013cf29a9e68e4411d0593040a8a1eb3-Abstract-Datasets_and_Benchmarks_Track.html), [arXiv:2406.09624](https://arxiv.org/abs/2406.09624), [code and downloads](https://github.com/Mohamedelrefaie/DrivAerNet), licence CC BY-NC 4.0, roughly 39 TB total, so subsetting is mandatory). For benchmark-design conventions see PDEBench ([arXiv:2210.07182](https://arxiv.org/abs/2210.07182)) and The Well ([arXiv:2412.00568](https://arxiv.org/abs/2412.00568)).

**Benchmarking.** Rabeh, Herron, Balu, Sarkar, Hegde, Krishnamurthy and Ganapathysubramanian, *Benchmarking scientific machine-learning approaches for flow prediction around complex geometries*, Communications Engineering 2025 ([DOI 10.1038/s44172-025-00513-3](https://www.nature.com/articles/s44172-025-00513-3)), compare neural operators against vision-transformer foundation models, study SDF versus binary mask encodings, find foundation models substantially better when data is limited, and report OOD generalization as unsolved across the board. Closest prior art, and the natural framing anchor. Vinuesa and Brunton survey what ML can and cannot do for CFD ([arXiv:2110.02085](https://arxiv.org/abs/2110.02085)).

**Uncertainty.** Deep ensembles are the standard scalable baseline ([Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017, arXiv:1612.01474](https://arxiv.org/abs/1612.01474)). Split conformal prediction gives finite-sample marginal coverage under exchangeability ([Angelopoulos and Bates, arXiv:2107.07511](https://arxiv.org/abs/2107.07511)); conformalized quantile regression adapts interval width locally ([Romano, Patterson, Candes, arXiv:1905.03222](https://arxiv.org/abs/1905.03222)).

**The gap.** No existing paper, to our reading, combines force self-consistency, symmetry and boundary residuals, multi-axis OOD splits, conformal coverage on both fields and coefficients, data-efficiency curves, and a solver-verified active learning loop into one reproducible protocol spanning a 2D and a 3D open aerodynamic dataset.

---

## 4. Contribution and novelty claims

**C1: a consistency protocol.** Cheap, model-agnostic diagnostics computed identically for every model: force self-consistency (integrated versus directly predicted coefficients), no-slip and divergence residuals, and a mirror-symmetry test.

**C2: multi-axis OOD under matched budgets.** Three geometry-conditioning families at equal parameter and step budgets, on Reynolds extrapolation, angle-of-attack extrapolation, and shape-family holdout (NACA 5 digit on AirfRANS; estateback or a wheel and underbody configuration on DrivAerNet++).

**C3: calibration under shift.** Ensembles plus split conformal, reporting coverage and interval width for pointwise surface pressure and for integrated lift and drag, in-distribution and per OOD split. In-distribution coverage holds by construction; the finding is how it degrades out of distribution and whether that differs by architecture.

**C4: a solver-verified active learning loop.** Uncertainty-driven acquisition proposes shape and condition pairs, a handful of which are simulated by the author in Fluent with a documented, grid-independent 2D RANS setup. An external test of whether the model's uncertainty points at genuinely hard cases.

**C5: open artifacts.** Code, split-ID manifests, model cards, and a small set of verified Fluent cases with meshes, journals and results.

**Threats to novelty.** The Communications Engineering benchmark already compares operators and foundation models across geometry encodings; our axis is self-consistency and calibration rather than accuracy ranking, so we cite it as motivation, not as a competitor. AirfRANS already ships Reynolds and AoA splits, which helps: we inherit credible splits and add shape-family holdout and diagnostics. Conformal prediction for PDE surrogates may appear mid-project; our differentiator is the pairing with force integration and the solver-verified loop. A stronger geometry model landing mid-project is harmless, since the protocol is model-agnostic and the model becomes another baseline. The Fluent set is tiny, and we will say so: its value is independence from the training-data generator, not statistical power.

**Why this is publishable even if results are mixed.** The claim is about a measurement protocol, not a leaderboard position. Findings of the form "the best-L2 model has the worst force consistency", "conformal intervals lose coverage sharply under shape-family shift but hold under mild Reynolds shift", or "data-efficiency curves cross, so ranking depends on training budget" are all publishable and more useful to a practitioner than another few percent of error reduction. The risk is presenting these as failures of the models rather than characterization of them, so the framing must stay diagnostic throughout.

---

## 5. Mathematical and algorithmic formulation

### 5.1 Operator learning setup

Let $\Omega_g \subset \mathbb{R}^d$ ($d=2$ for AirfRANS, $d=3$ for DrivAerNet++) be the fluid domain exterior to a body with boundary $\Gamma_g$, and let $c$ be the flow condition, for AirfRANS $c=(U_\infty,\alpha)$ with $\mathrm{Re}=U_\infty L/\nu$. We seek

$$
\mathcal{S}:(g,c)\longmapsto u=(\mathbf{v},p,\nu_t)\big|_{\Omega_g},
$$

restricted in practice to the surface trace $\mathcal{S}_\Gamma:(g,c)\mapsto (c_p,\boldsymbol{\tau}_w)\big|_{\Gamma_g}$, since surface pressure and wall shear stress are what forces depend on. AirfRANS also supplies volume fields, so there we train both and compare. A neural operator approximates $\mathcal{S}$ as lifting, $L$ kernel-integral layers and projection,

$$
\mathcal{S}_\theta=\mathcal{Q}\circ\sigma(\mathcal{K}_L+W_L)\circ\cdots\circ\sigma(\mathcal{K}_1+W_1)\circ\mathcal{P},
\qquad (\mathcal{K}_\ell h)(x)=\int_D \kappa_\ell(x,y)h(y)\,\mathrm{d}\mu(y),
$$

which in FNO becomes a truncated Fourier multiplier $\mathcal{K}_\ell h=\mathcal{F}^{-1}(R_\ell\cdot\mathcal{F}h)$.

### 5.2 Geometry encoding choices

**M1, graph baseline.** Node features $(x_i,\mathbf{n}_i,\kappa_i,c)$ with outward normal and local curvature on a k-nearest-neighbour graph, with $T$ message-passing rounds:

$$
m_{ij}^{(t)}=\phi_e\!\left(h_i^{(t)},h_j^{(t)},x_j-x_i,\|x_j-x_i\|\right),\qquad
h_i^{(t+1)}=\phi_v\!\left(h_i^{(t)},\textstyle\sum_{j\in\mathcal{N}(i)}m_{ij}^{(t)}\right).
$$

**M2, SDF-conditioned FNO (GINO style).** With $s_g(x)=\pm\min_{y\in\Gamma_g}\|x-y\|$ on a regular latent grid, a graph-kernel encoder transfers surface features to the grid, an FNO stack acts there, and a graph-kernel decoder queries back to arbitrary surface points:

$$
h_{\text{lat}}(x)=\int_{B_r(x)\cap\Gamma_g}\kappa_{\text{enc}}\!\left(x,y,s_g(x)\right)a(y)\,\mathrm{d}y,
\qquad
\hat u(z)=\int_{B_r(z)}\kappa_{\text{dec}}(z,x)\,h^{(L)}_{\text{lat}}(x)\,\mathrm{d}x .
$$

Discretization invariance holds because both transfers are quadrature approximations of continuum integrals.

**M3, geometry transformer (Transolver style).** Soft-assign $N$ mesh points to $M\ll N$ slices, attend among slice tokens, scatter back:

$$
w_{im}=\frac{\exp\langle q_m,h_i\rangle}{\sum_{m'}\exp\langle q_{m'},h_i\rangle},\qquad
z_m=\frac{\sum_i w_{im}h_i}{\sum_i w_{im}},\qquad
h_i'=\sum_m w_{im}\tilde z_m,
$$

at cost $\mathcal{O}(NM+M^2)$ rather than $\mathcal{O}(N^2)$. For M2 we ablate three conditioning inputs, SDF, binary occupancy mask, and SDF plus normals, mirroring the encoding comparison in the Communications Engineering benchmark.

### 5.3 Loss functions

Base loss, relative L2 per sample: $\mathcal{L}_{\text{data}}=\frac{1}{B}\sum_b \|\hat u_b-u_b\|_2 / (\|u_b\|_2+\varepsilon)$. Full objective, every extra term ablated on and off:

$$
\mathcal{L}=\mathcal{L}_{\text{data}}+\lambda_F\mathcal{L}_{\text{force}}+\lambda_B\mathcal{L}_{\text{bc}}+\lambda_S\mathcal{L}_{\text{sym}}+\lambda_D\mathcal{L}_{\text{div}} .
$$

Each term is normalized by its value at initialization so all start at order one; $\lambda$ is then swept over $\{0,0.01,0.1,1\}$ on the airfoil case only.

### 5.4 Force integration from surface fields

$$
\mathbf{F}=\oint_{\Gamma_g}\left(-p\,\mathbf{n}+\boldsymbol{\tau}_w\right)\mathrm{d}S
\;\approx\;\sum_{i=1}^{N_f}\left(-p_i\mathbf{n}_i+\boldsymbol{\tau}_{w,i}\right)\Delta S_i ,
$$

with $\Delta S_i$ the facet area (length in 2D), using the same quadrature the CFD post-processor uses. With freestream direction $\mathbf{e}_\infty$ and lift direction $\mathbf{e}_\perp$,

$$
C_D^{\text{int}}=\frac{\mathbf{F}\cdot\mathbf{e}_\infty}{\tfrac12\rho U_\infty^2 A_{\text{ref}}},\qquad
C_L^{\text{int}}=\frac{\mathbf{F}\cdot\mathbf{e}_\perp}{\tfrac12\rho U_\infty^2 A_{\text{ref}}} .
$$

Since $\Delta S_i$ and $\mathbf{n}_i$ come from the fixed input geometry, this map is linear in the predicted surface fields and differentiable, so $C^{\text{int}}$ can enter the loss at negligible cost. With $C^{\text{head}}$ from a direct regression head on the same backbone, define

$$
\mathrm{FSC}=\left|C^{\text{int}}-C^{\text{head}}\right|,\qquad
\mathrm{FSC}_{\text{rel}}=\frac{\left|C^{\text{int}}-C^{\text{head}}\right|}{\left|C^{\text{true}}\right|+\varepsilon}.
$$

FSC needs no ground truth, so it is computable on OOD splits and on unlabelled proposed geometries, which is what makes it usable inside the active learning loop. Training term: $\mathcal{L}_{\text{force}}=|C^{\text{int}}-C^{\text{true}}|+|C^{\text{head}}-C^{\text{true}}|+|C^{\text{int}}-C^{\text{head}}|$.

### 5.5 Consistency residuals

**No-slip.** For the volume model, with $\mathcal{N}_\delta$ the sample points within $\delta$ of the wall and $\omega(x)=1-\mathrm{dist}(x,\Gamma_g)/\delta$:
$\mathcal{L}_{\text{bc}}=\frac{1}{|\mathcal{N}_\delta|}\sum_{x\in\mathcal{N}_\delta}\|\hat{\mathbf{v}}(x)\|^2\omega(x)$.

**Divergence.** $\mathcal{L}_{\text{div}}=\frac{1}{|\mathcal{X}|}\sum_{x}\left(\nabla\cdot\hat{\mathbf{v}}(x)\right)^2$, by finite differences on the latent grid or least-squares gradients on the point cloud. Report the raw residual as a diagnostic even when $\lambda_D=0$.

**Symmetry.** With $R$ the reflection about the relevant plane and $R^\sharp$ its action on vector fields, for $R\Gamma_g=\Gamma_g$ and a symmetric condition (symmetric NACA at $\alpha=0$; mirror-symmetric car at zero yaw),

$$
\mathcal{L}_{\text{sym}}=\big\|\hat u(g,c)-R^\sharp\hat u(Rg,Rc)\big\|_2\big/\big\|\hat u(g,c)\big\|_2 .
$$

Equivariant architectures should score near zero; anything else measures reliance on augmentation and coordinate accidents. Also check antisymmetry on a symmetric airfoil at $\pm\alpha$: $C_L(-\alpha)\approx-C_L(\alpha)$, $C_D(-\alpha)\approx C_D(\alpha)$.

### 5.6 Deep ensembles and conformal prediction

Train $K=5$ members with independent initializations and shuffling:

$$
\bar u(x)=\frac1K\sum_k \hat u_{\theta_k}(x),\qquad
\hat\sigma^2(x)=\frac{1}{K-1}\sum_k\left(\hat u_{\theta_k}(x)-\bar u(x)\right)^2 ,
$$

propagating through the linear integration for coefficients, $\bar C_D=\frac1K\sum_k C_D^{\text{int}}[\hat u_{\theta_k}]$. On a calibration set of $n$ simulations disjoint from train and test, define the normalized nonconformity score $s_j=|y_j-\bar u(x_j)|/(\hat\sigma(x_j)+\varepsilon)$ and let $\hat q$ be its $\lceil (n+1)(1-\alpha)\rceil/n$ empirical quantile. Then

$$
\mathcal{C}_\alpha(x)=\left[\bar u(x)-\hat q\,\hat\sigma(x),\;\bar u(x)+\hat q\,\hat\sigma(x)\right]
$$

satisfies $\mathbb{P}(y\in\mathcal{C}_\alpha(x))\ge 1-\alpha$ when calibration and test data are exchangeable. Two granularities are calibrated separately: per-simulation coefficient intervals, where the exchangeable unit is a whole simulation and the guarantee is clean, and pointwise field intervals, where points within a simulation are not exchangeable, so we conformalize at the simulation level with a max-over-points or quantile-over-points score and state the weaker interpretation explicitly rather than pretend a pointwise guarantee holds. Reported: coverage $\widehat{\mathrm{Cov}}=\frac1m\sum_i\mathbb{1}[y_i\in\mathcal{C}_\alpha(x_i)]$ at nominal $1-\alpha\in\{0.8,0.9,0.95\}$, interval width, reliability curves, expected calibration error.

### 5.7 OOD split definitions

In-distribution: AirfRANS `full` random split and a random split of the DrivAerNet++ subset. Scarce: the `scarce` task. Reynolds shift: the `reynolds` task, error plotted against distance from the training band. AoA shift: the `aoa` task, broken out by pre-stall versus post-stall using a wall-shear sign-change indicator from ground truth. Shape-family shift (new): train NACA 4 digit, test NACA 5 digit; on DrivAerNet++, train fastback plus notchback and test estateback, and separately hold out one wheel or underbody configuration. Combined shift (new): shape family and Reynolds together, the hardest cell. For each split define a shift magnitude $\Delta$, the normalized distance of the test condition from training support, and plot error and coverage against $\Delta$. That plot is the paper's central figure.

### 5.8 Active learning acquisition

The pool is a parametric family of NACA 4 and 5 digit airfoils crossed with a grid of $(\mathrm{Re},\alpha)$, none in the training set. Score:

$$
a(g,c)=\underbrace{\hat\sigma_{C_D}(g,c)}_{\text{ensemble spread}}
+\beta\underbrace{\mathrm{FSC}(g,c)}_{\text{self-inconsistency}}
+\gamma\underbrace{\mathcal{L}_{\text{sym}}(g,c)}_{\text{symmetry violation}},
$$

all three normalized to unit variance over the pool, with $\beta=\gamma=1$ by default plus a sensitivity check. Select top-$k$ subject to greedy farthest-point diversity selection in normalized design space so we do not pick $k$ near-duplicates. Baselines: random selection and pure ensemble-variance selection. Selected cases run in Fluent (steady incompressible RANS, Spalart-Allmaras and $k$-$\omega$ SST, C-grid with documented $y^+$ target, three-level grid-independence study on one representative case). We report (i) whether high-acquisition cases really do have higher surrogate error than random ones, the falsifiable claim, and (ii) the solver-to-solver offset between our Fluent setup and the OpenFOAM setup behind AirfRANS, measured on a few replicated test cases. Point (ii) is essential: without it any disagreement is confounded by solver and turbulence-model differences.

---

## 6. Data, tools and compute

**AirfRANS.** 1000 simulations, roughly 32000 mesh points each, licence ODbL-1.0 (attribution and share-alike on derived databases; check before redistributing a processed copy). Access via the `airfrans` package ([PyPI](https://pypi.org/project/airfrans/)), the [library repo](https://github.com/Extrality/airfrans_lib) with direct download links, or PyTorch Geometric. Use the preprocessed cropped version, not the raw OpenFOAM one. **Verify the exact download size before committing a Colab session to it.** Small enough that the whole study runs on it end to end.

**DrivAerNet++.** Roughly 8000 designs, over 39 TB total, CC BY-NC 4.0 (non-commercial research only; fine for a preprint, state it in the paper), on Harvard Dataverse via Globus, with splits and drag and frontal-area CSVs in the [repo](https://github.com/Mohamedelrefaie/DrivAerNet). Subset strategy: download only coefficient CSVs plus the point-cloud or decimated-mesh modality, skipping volumetric fields; target 400 to 800 designs stratified across fastback, notchback and estateback and across wheel and underbody configurations so shape-family holdout stays possible; decimate to 8k to 16k points per design, float16 where precision allows, putting 800 designs in the low single-digit GB range; make the subsetting script reproducible from a seed and commit the exact design IDs, since that manifest is what lets others reproduce the study without 39 TB. If bandwidth blocks, drop to 200 designs and reframe the 3D part as a smaller transfer study. The 2D story stands alone regardless.

**Software.** Python 3.11, PyTorch 2.x, PyTorch Geometric, `neuraloperator` for FNO and GINO components ([https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)), the [Transolver reference code](https://github.com/thuml/Transolver), `airfrans`, trimesh or PyVista for geometry and normals, Hydra or YAML for sweeps, Weights and Biases free tier or CSV logging, Matplotlib. Ansys Fluent (student or university licence) plus Fluent Meshing or ICEM CFD, automated with TUI journals so the CFD pipeline is scripted and reproducible.

**Compute.** Designed for a single T4 or P100 on Colab or Kaggle, plus an optional 100 to 200 USD of spot L4 or A10G time for 3D.

| Stage | Hardware | Estimated wall-clock |
|---|---|---|
| AirfRANS single-model training (surface, 800 train cases) | Kaggle P100 | 2 to 5 hours |
| One 5-member ensemble, one architecture | Kaggle P100 | 10 to 25 hours across sessions |
| 3 architectures x 5 members x 5 splits, with checkpoint reuse | Kaggle + Colab | 3 to 4 weeks of background sessions |
| Data-efficiency sweep (6 sizes, 3 seeds, 1 architecture) | Kaggle P100 | 1 week of background sessions |
| DrivAerNet++ subset, 3 architectures | spot L4, 20 to 40 GPU-hours | 60 to 120 USD |
| Fluent verification (6 to 12 cases plus grid study) | local CPU | 1 to 3 days |

These are planning estimates, not measurements. Log actual times and report them: honest cost accounting is itself a contribution for a small-compute study. Every training script must checkpoint per epoch and resume cleanly, because free-tier sessions get killed.

---

## 7. Work plan

**Week 1, day by day.**

- *Day 1:* Repo, licence, README skeleton, environment file. Install `airfrans`, download the preprocessed dataset, verify size and checksum, record the real size in the README.
- *Day 2:* Data loader. Per simulation, extract surface points, normals, facet areas, $c_p$, $\boldsymbol{\tau}_w$, flow condition, and ground-truth $C_L$ and $C_D$. Cache to compact `.npz` or `.pt`.
- *Day 3:* Implement force integration and validate it against the dataset's own reported coefficients on ground-truth fields. This must agree closely before anything else proceeds; if it does not, the quadrature or reference-area convention is wrong. The most important day of week 1.
- *Day 4:* Implement and train M1 on `full`. Get any non-trivial number on the board.
- *Day 5:* Evaluation harness: relative L2 per field, coefficient errors, FSC, symmetry residual, per-split results JSON in a fixed schema.
- *Day 6:* Wire up M2 from `neuraloperator` components; train briefly to confirm the pipeline runs end to end.
- *Day 7:* Freeze the config system and results schema. Write the arXiv skeleton with every heading and every figure as a placeholder. Commit. No schema changes after this without a migration script.

**Weeks 2 to 3, three models in-distribution.** Bring M3 online. Tune fairly: parameter budgets within roughly 20 percent of each other, same optimizer family and step budget, learning rate swept over three values per model. Milestone: Table 1 complete.

**Weeks 4 to 5, OOD.** All three models on `scarce`, `reynolds`, `aoa`, and the new shape-family and combined splits. Milestone: the error-versus-shift figure exists and the central claim is either visible in it or not. If not, pivot emphasis to calibration.

**Weeks 5 to 6, uncertainty.** Five-member ensembles per model, reusing earlier runs as members where seeds already differ. Split conformal at both granularities. Milestone: pilot results good enough for a 4-page workshop submission.

**Week 7, data efficiency.** Training-set sizes of 25, 50, 100, 200, 400 and 800, three seeds, all three models on the airfoil case.

**Weeks 7 to 9, DrivAerNet++ subset.** Download, subset, retrain the same three models on surface pressure and drag, repeat consistency, shape-family OOD and calibration. Reduced scope acceptable here.

**Weeks 9 to 10, active learning and Fluent verification.** Build and score the pool, select with three acquisition strategies, run 6 to 12 Fluent cases plus grid independence and the solver-offset calibration, evaluate.

**Weeks 10 to 12, writing and release.** Draft, figure freeze, code cleanup, model cards, Zenodo archive, arXiv submission. Reserve the last week entirely for writing.

---

## 8. Experiments and evaluation

**Metrics.** *Field:* relative L2 on $c_p$ and each component of $\boldsymbol{\tau}_w$, mean absolute error, near-wall-masked error. *Coefficients:* absolute and relative error on $C_L$ and $C_D$ from both integration and head, plus Spearman rank correlation on $C_D$, which is what matters for design ranking. *Consistency:* FSC and FSC-relative, symmetry residual, the antisymmetry check, divergence residual (volume model), no-slip residual. *Uncertainty:* coverage at nominal 80, 90 and 95 percent, mean and median interval width, reliability-curve area, and the ratio of OOD coverage to in-distribution coverage, the single number summarizing calibration robustness. *Cost:* parameters, training GPU-hours, inference latency, peak memory.

**Baselines.** (a) Constant predictor (dataset-mean field and mean $C_D$), anchoring what learning nothing scores. (b) Ridge regression from geometry parameters and flow condition directly to $C_D$, a shockingly strong baseline on parametric datasets that must be reported honestly. (c) M1. (d) M2. (e) M3. (f) Where AirfRANS published baselines are directly comparable, quote them with attribution rather than claim reproduction.

**Ablations.** Geometry encoding (SDF versus mask versus SDF plus normals) on M2. Physics-loss weights on and off. Ensemble size $K\in\{1,3,5,10\}$ against calibration quality. Conformal score choice (absolute residual, ensemble-normalized, quantile-regression-based). Surface-only versus volume-then-trace on AirfRANS. Rotation and reflection augmentation on and off, which interacts directly with the symmetry residual.

**Figures.** 1. Schematic of the three families with the shared protocol beneath. 2. $c_p$ distributions on 4 airfoils, prediction versus truth with ensemble interval shaded, one in-distribution and three OOD. 3. Surface pressure on 3 cars: prediction, truth, signed error. 4. Error versus shift magnitude, one panel per OOD axis, one line per model. 5. Force self-consistency scatter, $C_D^{\text{int}}$ against $C_D^{\text{head}}$, coloured by split, with identity line. 6. Reliability diagram, empirical versus nominal coverage, per split, ensemble-only versus conformalized. 7. Coverage versus shift magnitude, showing where exchangeability stops holding. 8. Interval-width distributions, in-distribution against OOD. 9. Data-efficiency curves, log-log, with seed bands. 10. Symmetry residual per model, with and without augmentation. 11. Active learning: surrogate error on Fluent-verified cases selected by uncertainty, by FSC, and at random. 12. Cost-accuracy Pareto plot, relative L2 against training GPU-hours.

**Tables.** 1. In-distribution accuracy, consistency and cost, all models, both datasets. 2. OOD results, models by splits, field and coefficient error. 3. Calibration: coverage and width at three nominal levels, by model and split. 4. Ablation on geometry encoding and physics-loss weights. 5. Fluent verification cases: geometry, conditions, mesh size, $y^+$, grid-independence result, solver offset, surrogate prediction and interval. 6. Dataset and split card: sizes, licences, exact subset IDs.

---

## 9. Expected results and interpretation

The most likely picture: all three models are accurate in-distribution, with differences between them smaller than differences between splits. That alone is worth stating clearly, because it implies the field's current model-comparison exercises measure noise relative to the deployment-relevant axis.

On OOD, expect a hierarchy. Mild Reynolds extrapolation should degrade gracefully; angle-of-attack extrapolation should degrade sharply once test cases cross into a flow regime, separation, absent from training; shape-family holdout should sit in between. If AoA failure correlates with separation onset rather than with numerical distance from the training range, that is a clean, quotable physical finding: these models interpolate within a flow regime, they do not extrapolate across one.

On consistency, the interesting hypothesis is that FSC is *not* strongly correlated with field L2. If a model with better L2 has worse force consistency, the standard metric is actively misleading for design use. Expect FSC to grow faster than field error under shift, making it a usable label-free OOD detector: the practical payoff of C1 and the input to C4.

On calibration, in-distribution coverage hitting nominal is a sanity check, not a finding. The finding is the OOD coverage curve. Expect substantial undercoverage under shape-family and post-stall shift, with ensemble-only intervals worse than conformalized ones in-distribution but similarly broken out of it. The honest conclusion, if that holds: conformal prediction fixes calibration but not robustness, and a conformal guarantee should not be read as protection against design-space extrapolation.

On data efficiency, expect the transformer to need more data and the graph baseline to hold up better when data is scarce, consistent with the reported finding that method ranking flips under scarcity. Crossing curves would be a good outcome.

**Negative results that remain publishable.** Physics-consistency losses fail to improve accuracy and only shrink the residual they penalize. Symmetry augmentation fixes the symmetry residual and nothing else. Active learning selects cases no harder than random ones, falsifying a widely assumed premise. The ridge baseline is competitive on $C_D$ ranking, an uncomfortable but important observation about parametric benchmark datasets. All are reportable provided the paper stays framed as a diagnostic study.

---

## 10. Risks and mitigations

**Compute overrun.** Free-tier sessions get killed and a 3 x 5 x 5 grid is many runs. Checkpoint every epoch and resume automatically; reuse existing seeds as ensemble members; make AirfRANS the self-sufficient paper and let DrivAerNet++ shrink; hold a rule that any run exceeding 6 hours gets its budget cut, not its deadline extended.

**Dataset size and access.** DrivAerNet++ is 39 TB and needs Globus. Verify access in week 1, not week 7; download only coefficients and decimated surfaces; commit the subset ID list; keep the 200-design fallback ready. Verify the AirfRANS download size before starting.

**Licensing.** AirfRANS is ODbL-1.0 with share-alike obligations on derived databases; DrivAerNet++ is CC BY-NC 4.0, non-commercial only. Release code and split-ID manifests rather than redistributing processed copies unless the licence clearly permits it, state both licences, and keep the new Fluent cases under a clean permissive licence so they are unencumbered.

**Confounded Fluent verification.** Our setup will not match the OpenFOAM setup behind AirfRANS in turbulence model, discretization, mesh topology or convergence criteria. Measure the offset by replicating a few AirfRANS test cases before interpreting any surrogate-versus-Fluent gap, and report $y^+$, grid independence and residual histories in an appendix.

**Overclaiming.** The biggest risk to credibility. With 6 to 12 verification cases we cannot claim active learning works; with one dataset family per dimension we cannot claim results generalize to all external aerodynamics. Mitigations: a dedicated Limitations section, claims phrased as "on these datasets, under these budgets", no headline claim resting on fewer than three seeds, and an explicit statement that the Fluent set is a qualitative external check rather than a statistical test.

**Scooping.** Post to arXiv as soon as the AirfRANS half is complete and defensible rather than waiting for the 3D half; preprint plus public repo establish the timestamp.

**Solo-author blind spots.** No co-author to catch a bug in force integration or a leak in the splits. Unit-test force integration against ground truth on day 3; assert zero simulation-ID overlap between train, calibration and test for every split; ask two reviewers, one CFD and one ML, before submission.

---

## 11. Venue plan with deadlines and paper outline

1. **arXiv, late November 2026.** The non-negotiable anchor. Submit the AirfRANS-complete version even if DrivAerNet++ is partial. cs.LG primary, physics.flu-dyn cross-list.
2. **NeurIPS 2026 ML4PS workshop, 4 pages, 12 September 2026.** Feasible only if a Phase 3 pilot exists by early September, meaning work starts by early July. If so, submit the consistency protocol plus in-distribution results and one OOD axis. If not, skip it without regret.
3. **AIAA AVIATION 2027, abstract mid-November 2026.** Strong fit: the CFD Vision 2030 track explicitly covers AI and ML for CFD, and that audience values exactly the consistency and verification angle ML venues undervalue. The Fluent component plays far better here. The abstract is cheap and aligns with the arXiv post.
4. **ICLR 2027 workshops, February 2027.** Natural home for the calibration and OOD story; submit a refined version once arXiv is up.
5. **Journal, first half of 2027.** Ranked by fit: *Data-Centric Engineering* (open, benchmark-friendly, likely fastest for an evaluation paper); *Machine Learning: Science and Technology* (good for the UQ framing); *Computers and Fluids* (good if the CFD verification is expanded); *Physics of Fluids* (fast, physics-forward); *CMAME* or *JCP* (highest prestige, hardest bar, would need the consistency protocol to mature into a training method with demonstrated gains). Recommendation: Data-Centric Engineering or MLST first.

**Paper outline, 8 to 10 pages plus appendix.** 1. Introduction framed on the three gaps. 2. Related work. 3. Problem setup and the consistency protocol, the conceptual core. 4. Models and training. 5. Datasets and splits. 6. Results: in-distribution, OOD, calibration, data efficiency. 7. Active learning with Fluent verification. 8. Discussion and limitations. 9. Conclusion. Appendices: hyperparameters, CFD setup and grid study, extra figures, compute accounting.

---

## 12. Which professors and programs this signals, and why

1. **Anima Anandkumar (Caltech, NVIDIA).** GINO is a direct baseline and SDF conditioning is her group's question; a rigorous, fair external audit is work they engage with.
2. **Kamyar Azizzadenesheli (NVIDIA).** GINO co-author focused on deploying operators for real engineering, which is what the cost-accuracy and calibration axes address.
3. **Nikola Kovachki (NVIDIA).** Operator theory and discretization invariance; the careful quadrature treatment in force integration and graph-kernel transfers is theory-adjacent in a way he would notice.
4. **George Em Karniadakis (Brown).** DeepONet and the question of what physics-informed learning actually guarantees; consistency residuals are core to his group.
5. **Lu Lu (Yale).** DeepONet co-author whose group works on operator benchmarking, error estimation and UQ. Unusually close methodological fit.
6. **Faez Ahmed (MIT DeCoDE).** DrivAerNet++ senior author, design-oriented, cares whether surrogates support real design decisions, which is exactly the FSC and calibration argument.
7. **Mohamed Elrefaie (MIT).** DrivAerNet++ first author; using his dataset thoughtfully, with a reproducible subset manifest and clean licence handling, is a strong basis for contact.
8. **Johannes Brandstetter (JKU Linz, Emmi AI).** Neural PDE surrogates with a public emphasis on rigorous evaluation and on what these models do versus what is claimed.
9. **Nils Thuerey (TU Munich).** Physics-based deep learning and long-standing scepticism about benchmark-driven claims; the negative-results component suits his group.
10. **Siddhartha Mishra (ETH Zurich).** Operator theory and error analysis on when neural operators generalize; the OOD characterization is the empirical counterpart.
11. **Amir Barati Farimani (CMU).** Transformers and geometry-aware learning for fluid and mechanical systems, plus interest in surrogate deployment.
12. **Romit Maulik (Penn State).** ROM, SciML and UQ, with explicit interest in whether uncertainty estimates are trustworthy. Strong overlap with the conformal component.
13. **Karthik Duraisamy (University of Michigan).** Data-driven turbulence modelling and a well-argued position on credibility and verification of ML-augmented CFD; the solver-offset analysis is written for a reader like him.
14. **Ricardo Vinuesa (KTH).** ML for fluid mechanics with emphasis on physical interpretability; the symmetry and consistency diagnostics fit his framing.
15. **Nathan Kutz and Steven Brunton (University of Washington).** Data-driven modelling and ROM with explicit interest in when reduced models fail, which the shift analysis and data-efficiency curves measure.
16. **Baskar Ganapathysubramanian (Iowa State) and Chinmay Hegde (NYU).** Senior authors of the Communications Engineering benchmark this project extends; the closest institutional match to its purpose.
17. **Patrick Gallinari and Paola Cinnella (Sorbonne).** AirfRANS senior authors spanning ML and turbulence modelling; a careful extension of their benchmark is a natural opener.

The combined signal is specific: this candidate reads operator-learning papers critically, implements three non-trivial architectures, reasons about quadrature and boundary conditions, runs and documents a real CFD verification study, and writes a paper that says what it does not know. For a mechanical engineering BSc applying to computational mechanics and SciML groups, the CFD-plus-ML combination is the differentiator, and this project puts it in the foreground.

---

## 13. Skills and artifacts produced

```
geo-operator-audit/
  README.md                 # results table, reproduction steps, licence notes
  LICENSE                   # MIT for code
  environment.yml
  configs/                  # one YAML per model x split, Hydra-composed
  src/
    data/                   # airfrans_loader.py, drivaernet_subset.py, splits.py
    geometry/               # sdf.py, normals.py, quadrature.py, symmetry.py
    models/                 # gnn.py, sdf_fno.py, geo_transformer.py, heads.py
    physics/                # force_integration.py, residuals.py
    uq/                     # ensembles.py, conformal.py, coverage.py
    active/                 # pool.py, acquisition.py, diversity.py
    eval/                   # metrics.py, harness.py, schema.py
  scripts/                  # train.py, evaluate.py, make_figures.py, subset_drivaernet.py
  fluent/                   # journals, mesh scripts, grid study, raw results, README
  tests/                    # test_force_integration.py, test_splits_no_leakage.py
  results/                  # per-run JSON in the schema frozen in week 1
  paper/                    # LaTeX source, figures
  model_cards/              # one markdown card per trained model
```

**Model cards.** One per architecture and dataset: intended use, training data and split, data licence, parameters, training GPU-hours, in-distribution and OOD metrics, calibration coverage, known failure modes (post-stall, shape-family shift), and an explicit "do not use for" statement.

**Released artifacts.** Code under MIT, split-ID manifests, model cards, checkpoints where size permits, and the Fluent verification set on Zenodo with a DOI: meshes, journals, convergence histories, extracted surface data.

**Skills demonstrated.** Operator learning from paper to code; geometry processing (SDF, normals, curvature, quadrature on unstructured surfaces); graph and attention architectures on constrained hardware; distribution-free UQ; experiment design under matched budgets; CFD setup, meshing, grid independence and verification in Fluent; reproducible research engineering; scientific writing.

**CV line.** *"Geometry-aware neural operator surrogates for external aerodynamics: physics-consistency, out-of-distribution generalization and calibrated uncertainty." T. Amin. arXiv preprint, 2026. Code and a solver-verified CFD test set released openly.*

---

## 14. Ordered reading list

Items 1 to 5 before writing code, 6 to 9 during weeks 1 to 3, the rest as the matching phase begins.

1. Bonnet, Mazari, Cinnella, Gallinari, *AirfRANS*, NeurIPS 2022. [https://arxiv.org/abs/2212.07564](https://arxiv.org/abs/2212.07564). Read the metric definitions and four task splits closely; they define half this project.
2. Rabeh et al., *Benchmarking scientific machine-learning approaches for flow prediction around complex geometries*, Communications Engineering 2025. [https://www.nature.com/articles/s44172-025-00513-3](https://www.nature.com/articles/s44172-025-00513-3). The framing anchor and closest prior art.
3. Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021. [https://arxiv.org/abs/2010.08895](https://arxiv.org/abs/2010.08895).
4. Li, Kovachki et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. [https://arxiv.org/abs/2309.00583](https://arxiv.org/abs/2309.00583). Read alongside the `neuraloperator` source.
5. Wu et al., *Transolver*, ICML 2024. [https://arxiv.org/abs/2402.02366](https://arxiv.org/abs/2402.02366) and [https://github.com/thuml/Transolver](https://github.com/thuml/Transolver).
6. Elrefaie, Morar, Dai, Ahmed, *DrivAerNet++*, NeurIPS 2024. [https://arxiv.org/abs/2406.09624](https://arxiv.org/abs/2406.09624). Read the tasks and split conventions before downloading anything.
7. Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces*, JMLR 2023. [https://arxiv.org/abs/2108.08481](https://arxiv.org/abs/2108.08481). Skim the theory, read discretization invariance carefully.
8. Lu, Jin, Pang, Zhang, Karniadakis, *DeepONet*, Nature Machine Intelligence 2021. [https://arxiv.org/abs/1910.03193](https://arxiv.org/abs/1910.03193).
9. Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021. [https://arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409). The M1 baseline's ancestry.
10. Lakshminarayanan, Pritzel, Blundell, *Deep Ensembles*, NeurIPS 2017. [https://arxiv.org/abs/1612.01474](https://arxiv.org/abs/1612.01474).
11. Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction*. [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511). Read all of it; it is the backbone of Section 5.6.
12. Romano, Patterson, Candes, *Conformalized Quantile Regression*, NeurIPS 2019. [https://arxiv.org/abs/1905.03222](https://arxiv.org/abs/1905.03222).
13. Li et al., *Physics-Informed Neural Operator*. [https://arxiv.org/abs/2111.03794](https://arxiv.org/abs/2111.03794). Relevant to the physics-loss ablation and to your existing PINO code.
14. Takamoto et al., *PDEBench*, NeurIPS 2022. [https://arxiv.org/abs/2210.07182](https://arxiv.org/abs/2210.07182). Benchmark-design conventions worth imitating.
15. Ohana et al., *The Well*, NeurIPS 2024. [https://arxiv.org/abs/2412.00568](https://arxiv.org/abs/2412.00568). How a modern large physics dataset documents itself.
16. Vinuesa and Brunton, *Enhancing computational fluid dynamics with machine learning*. [https://arxiv.org/abs/2110.02085](https://arxiv.org/abs/2110.02085). For the introduction's framing and for anticipating CFD-community objections.

---

## 15. Open questions for Taimoor

1. **Fluent licence and machine.** Do you have reliable Fluent access for the full 8 to 12 weeks, and enough local CPU for 6 to 12 2D RANS cases plus a three-level grid study? If not, the fallback is OpenFOAM in a container, which weakens the unique-angle claim but keeps the loop intact. Decide in week 0, not week 9.
2. **Which dataset carries the paper?** The safe plan makes AirfRANS self-sufficient with DrivAerNet++ as an extension. Will you ship with a partial 3D section if the download or GPU budget bites? Commit now, because it changes the week 7 decision.
3. **Three models or two?** Three architectures times five splits times five members is a large grid for free GPUs. Cutting to two and buying depth in calibration and active learning is probably the better trade.
4. **Timeline reality.** ML4PS on 12 September 2026 requires starting by early July. Feasible alongside your other commitments? If not, drop it explicitly and target the AIAA abstract plus arXiv plus an ICLR 2027 workshop.
5. **Volume or surface only on AirfRANS?** Surface-only is far cheaper and is what forces depend on, but divergence and no-slip residuals need the volume field. Worth the compute, or does the consistency story rest on force self-consistency and symmetry alone?
6. **How much do you want the physics-loss experiments?** Most likely part to yield a flat negative result. Publishable, but they eat compute; one clean ablation on one architecture may be enough.
7. **Which reviewer do you most want to satisfy?** An ML reviewer wants novelty and baselines; a CFD reviewer wants verification, $y^+$, grid independence and honest error bars. The introduction can lead with only one. Given where your applications go, which community should read it first?
8. **Co-reviewer.** Name two people now, one CFD and one ML, and ask them in week 6, not week 11.
