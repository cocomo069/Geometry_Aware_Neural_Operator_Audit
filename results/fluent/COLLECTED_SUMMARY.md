# Fluent campaign -- collected results

## Status tally (arm / model / variant)

| arm | model | variant | status | n |
|---|---|---|---|---|
| acquisition | sa | s0 | diverged | 8 |
| acquisition | sst | s0 | not_run | 8 |
| gridstudy | sa | s0 | converged | 3 |
| gridstudy | sst | s0 | converged | 3 |
| random | sa | s0 | converged | 4 |
| random | sa | s0 | diverged | 4 |
| random | sst | s0 | not_run | 8 |
| variance | sa | s0 | diverged | 8 |
| variance | sst | s0 | not_run | 8 |

## Accepted steady SA points (n=7)

| case | arm | variant | Re | aoa | status | CD | CL | y+max | orth |
|---|---|---|---|---|---|---|---|---|---|
| gridstudy_naca0012_re3e6_a5_L1 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01112 | 0.5538 | 0.87 | 0.265 |
| gridstudy_naca0012_re3e6_a5_L2 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01076 | 0.5530 | 0.59 | 0.271 |
| gridstudy_naca0012_re3e6_a5_L3 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01075 | 0.5521 | 0.39 | 0.274 |
| al_rand_naca0010_re7e6_a6 | random | s0 | 7e+06 | 6 | converged | 0.00979 | 0.6703 | 0.69 | 0.236 |
| al_rand_naca2415_re6e6_a0 | random | s0 | 6e+06 | 0 | converged | 0.00946 | 0.2245 | 0.38 | 0.287 |
| al_rand_naca33012_re6e6_am6 | random | s0 | 6e+06 | -6 | converged | 0.01020 | -0.4718 | 0.77 | 0.186 |
| al_rand_naca4415_re2e6_a0 | random | s0 | 2e+06 | 0 | converged | 0.01167 | 0.4296 | 0.41 | 0.281 |

## Grid-study GCI (Celik 2008)

### SA  (r21=1.500, r32=1.513, levels_converged=True)
- CD: p_obs=17.362, GCI_fine=0.000%, phi_ext=0.010755
- CL: p_obs=0.140, GCI_fine=3.390%, phi_ext=0.537126

### SST  (r21=1.500, r32=1.513, levels_converged=True)
- CD: p_obs=3.347, GCI_fine=0.257%, phi_ext=0.010684  [OSCILLATORY]
- CL: p_obs=1.864, GCI_fine=0.218%, phi_ext=0.543986

