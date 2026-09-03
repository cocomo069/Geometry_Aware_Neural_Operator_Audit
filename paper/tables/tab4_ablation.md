### M2 geometry conditioning (field p rel-L2)

| conditioning | full | shape5 (OOD) |
| --- | --- | --- |
| SDF (baseline) | 0.0257 | 0.06414 |
| binary mask | 0.02935 | 0.07279 |
| SDF + normals | 0.03092 | 0.06127 |

### M1 physics/force-consistency loss

| GNN variant | full p rel-L2 | full FSC | combined p rel-L2 | combined FSC |
| --- | --- | --- | --- | --- |
| baseline (λ_F=0) | 0.07374 | 0.0053 | 0.2697 | 0.01761 |
| + force loss | 0.1043 | 0.001333 | 0.259 | 0.005926 |

