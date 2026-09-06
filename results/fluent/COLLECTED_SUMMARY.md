# Fluent campaign -- collected results

## Status tally (arm / model / variant)

| arm | model | variant | status | n |
|---|---|---|---|---|
| acquisition | sa | r1 | diverged | 8 |
| acquisition | sa | s0 | diverged | 8 |
| acquisition | sst | s0 | not_run | 8 |
| gridstudy | sa | s0 | converged | 3 |
| gridstudy | sst | s0 | converged | 3 |
| offset | sa | s0 | converged | 5 |
| offset | sa | s0 | quasi_steady | 1 |
| offset | sst | s0 | converged | 6 |
| random | sa | r1 | diverged | 2 |
| random | sa | r1 | quasi_steady | 2 |
| random | sa | s0 | converged | 4 |
| random | sa | s0 | diverged | 4 |
| random | sst | s0 | not_run | 8 |
| variance | sa | r1 | converged | 1 |
| variance | sa | r1 | diverged | 7 |
| variance | sa | s0 | diverged | 8 |
| variance | sst | s0 | not_run | 8 |

## Accepted steady SA points (n=16)

| case | arm | variant | Re | aoa | status | CD | CL | y+max | orth |
|---|---|---|---|---|---|---|---|---|---|
| gridstudy_naca0012_re3e6_a5_L1 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01112 | 0.5538 | 0.87 | 0.265 |
| gridstudy_naca0012_re3e6_a5_L2 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01076 | 0.5530 | 0.59 | 0.271 |
| gridstudy_naca0012_re3e6_a5_L3 | gridstudy | s0 | 3e+06 | 5 | converged | 0.01075 | 0.5521 | 0.39 | 0.274 |
| offset_1_naca0016_a2p4 | offset | s0 | 3.9e+06 | 2.4 | converged | 0.01048 | 0.2609 | 0.43 | 0.295 |
| offset_2_naca0109_a8p6 | offset | s0 | 4e+06 | 8.58 | converged | 0.01301 | 0.9411 | 0.96 | 0.216 |
| offset_3_naca2211_a3p3 | offset | s0 | 4.6e+06 | 3.33 | converged | 0.00992 | 0.6245 | 0.45 | 0.282 |
| offset_4_naca2710_am1p6 | offset | s0 | 2.1e+06 | -1.58 | converged | 0.00972 | 0.0635 | 0.49 | 0.288 |
| offset_5_naca0010_a3p8 | offset | s0 | 6e+06 | 3.79 | converged | 0.00881 | 0.4259 | 0.56 | 0.256 |
| offset_6_naca5_17_a0p9 | offset | s0 | 4.1e+06 | 0.932 | quasi_steady | 0.01082 | 0.2298 | 0.42 | 0.295 |
| al_rand_naca0010_re7e6_a6 | random | s0 | 7e+06 | 6 | converged | 0.00979 | 0.6703 | 0.69 | 0.236 |
| al_rand_naca22012_re3e6_a12 | random | r1 | 3e+06 | 12 | quasi_steady | 0.01943 | 1.3575 | 0.85 | 0.201 |
| al_rand_naca2415_re6e6_a0 | random | s0 | 6e+06 | 0 | converged | 0.00946 | 0.2245 | 0.38 | 0.287 |
| al_rand_naca33012_re6e6_am6 | random | s0 | 6e+06 | -6 | converged | 0.01020 | -0.4718 | 0.77 | 0.186 |
| al_rand_naca4412_re7e6_a18 | random | r1 | 7e+06 | 18 | quasi_steady | 0.11432 | 1.7169 | 1.00 | 0.191 |
| al_rand_naca4415_re2e6_a0 | random | s0 | 2e+06 | 0 | converged | 0.01167 | 0.4296 | 0.41 | 0.281 |
| al_var_naca32012_re6e6_a18 | variance | r1 | 6e+06 | 18 | converged | 0.16624 | 1.9933 | 1.11 | 0.137 |

## Grid-study GCI (Celik 2008)

### SA  (r21=1.500, r32=1.513, levels_converged=True)
- CD: p_obs=17.362, GCI_fine=0.000%, phi_ext=0.010755
- CL: p_obs=0.140, GCI_fine=3.390%, phi_ext=0.537126

### SST  (r21=1.500, r32=1.513, levels_converged=True)
- CD: p_obs=3.347, GCI_fine=0.257%, phi_ext=0.010684  [OSCILLATORY]
- CL: p_obs=1.864, GCI_fine=0.218%, phi_ext=0.543986

