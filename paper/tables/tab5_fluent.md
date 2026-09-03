# Table 5 -- Fluent verification (a selection+outcome, b offset, c verified)

Caption: grid-converged at L2 (GCI_fine CD(SA)=0.00%); n = 6, qualitative external check. Surrogate CD is the input-robust cd_head.

## 5a Acquisition selection + steady outcome
| case | NACA | Re | alpha | S0 | r1 | surrogate CD_head |
| --- | --- | --- | --- | --- | --- | --- |
| naca0010_re3e6_a18 | 0010 | 3e+06 | 18 | diverged | not_run | 0.04079 $\pm$ 0.0023 |
| naca0010_re7e6_a18 | 0010 | 7e+06 | 18 | diverged | not_run | 0.03122 $\pm$ 0.003 |
| naca0015_re6e6_a18 | 0015 | 6e+06 | 18 | diverged | not_run | 0.03166 $\pm$ 0.0022 |
| naca24015_re7e6_a18 | 24015 | 7e+06 | 18 | diverged | not_run | 0.0323 $\pm$ 0.0022 |
| naca2410_re5e6_a18 | 2410 | 5e+06 | 18 | diverged | not_run | 0.03739 $\pm$ 0.0011 |
| naca33012_re7e6_a18 | 33012 | 7e+06 | 18 | diverged | not_run | 0.03252 $\pm$ 0.0023 |
| naca4410_re7e6_a18 | 4410 | 7e+06 | 18 | diverged | not_run | 0.0349 $\pm$ 0.0024 |
| naca4415_re7e6_a18 | 4415 | 7e+06 | 18 | diverged | not_run | 0.03498 $\pm$ 0.0018 |

## 5b Solver offset (6 AirfRANS replicas)
| replica | CD_AF | CL_AF | CD_SA | CD_SST | dCD | rel dCD | dCL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offset_4_naca2710_am1p6 | 0.009226 | 0.0513 | 0.009719 | 0.009553 | 0.000493 | 0.0534 | 0.0122 |
| offset_1_naca0016_a2p4 | 0.009702 | 0.251 | 0.01048 | 0.01019 | 0.00078 | 0.0804 | 0.00997 |
| offset_2_naca0109_a8p6 | 0.01151 | 0.926 | 0.01301 | 0.01304 | 0.0015 | 0.13 | 0.0152 |
| offset_6_naca5_17_a0p9 | 0.009805 | 0.219 | 0.01082 | -- | 0.00101 | 0.103 | 0.0112 |
| offset_3_naca2211_a3p3 | 0.009052 | 0.607 | 0.009919 | 0.009786 | 0.000867 | 0.0958 | 0.0175 |
| offset_5_naca0010_a3p8 | 0.008058 | 0.419 | 0.00881 | 0.008599 | 0.000752 | 0.0933 | 0.00729 |
- SA: dbar_CD=+0.0009006 (s=0.00034), dbar_CL=+0.0122 (s=0.0037), n=6
- SST: dbar_CD=+0.0007248 (s=0.00047), dbar_CL=+0.00793 (s=0.0057), n=5

## 5c Verified surrogate-vs-Fluent comparison (accepted set)
| case | arm | NACA | Re | alpha | status | CD_fl | CD_fl-dbar | Transolver | SDF-FNO | GNN | cov90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| al_rand_naca0010_re7e6_a6 | random | 0010 | 7e+06 | 6 | converged | 0.009794 | 0.009014 | 0.0105$\pm$0.0004 | -- | -- | False |
| al_rand_naca2415_re6e6_a0 | random | 2415 | 6e+06 | 0 | converged | 0.009461 | 0.00868 | 0.00923$\pm$0.0002 | -- | -- | False |
| al_rand_naca33012_re6e6_am6 | random | 33012 | 6e+06 | -6 | converged | 0.0102 | 0.009421 | 0.009883$\pm$0.0005 | -- | -- | False |
| al_rand_naca4415_re2e6_a0 | random | 4415 | 2e+06 | 0 | converged | 0.01167 | 0.01089 | 0.01272$\pm$0.0006 | -- | -- | False |
| gridstudy_naca0012_re3e6_a5_L2 | gridstudy | 0012 | 3e+06 | 5 | converged | 0.01076 | 0.009975 | 0.01199$\pm$0.0005 | -- | -- | False |
| offset_1_naca0016_a2p4 | offset | 0016 | 3.9e+06 | 2.4 | converged | 0.01048 | 0.009702 | 0.0097$\pm$4e-06 | -- | -- | True |

