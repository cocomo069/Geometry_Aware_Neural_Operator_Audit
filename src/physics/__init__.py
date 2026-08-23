"""Physics: force integration and consistency residuals (A2)."""
from src.physics.force_integration import integrate_forces
from src.physics.residuals import (
    divergence_residual,
    knn_ls_weights,
    noslip_residual,
    pointwise_divergence,
)
