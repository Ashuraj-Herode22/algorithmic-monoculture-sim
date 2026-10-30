"""
Unit Tests for Mathematical Identities and Core Modules.
Verifies Page's Theorem equality to machine precision and matrix stochasticity.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root and src are available on path
ROOT_PATH = str(Path(__file__).resolve().parent.parent)
SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
for p in (ROOT_PATH, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.metrics import compute_page_decomposition, compute_shannon_entropy, compute_monoculture_index
from src.networks import generate_watts_strogatz_graph, generate_barabasi_albert_graph, get_degroot_trust_matrix
from src.agents import initialize_human_beliefs, generate_correlated_ai_advice, execute_hybrid_degroot_step


def test_page_diversity_theorem():
    """Verifies (c - theta)^2 == avg_individual_error - cognitive_diversity to 1e-12 tolerance."""
    truth = 100.0
    rng = np.random.default_rng(42)
    estimates = rng.normal(loc=102.0, scale=15.0, size=500)

    results = compute_page_decomposition(estimates, truth)
    residual = results["identity_residual"]

    assert residual < 1e-12, f"Page's Theorem identity violated! Residual: {residual}"
    print(f"✓ Page's Diversity Prediction Theorem verified to residual: {residual:.2e}")


def test_degroot_matrix_stochasticity():
    """Verifies that each row of W sums to exactly 1.0."""
    G = generate_watts_strogatz_graph(n=100, k=4, p=0.1, seed=42)
    W = get_degroot_trust_matrix(G, self_weight=0.5)

    row_sums = np.sum(W, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
    print("✓ DeGroot trust matrix row-stochasticity verified (all rows sum to 1.0).")


def test_hybrid_degroot_boundary_conditions():
    """Verifies behavior at alpha = 0 (pure social) and alpha = 1 (pure AI)."""
    n = 50
    beliefs = np.ones(n) * 10.0
    ai_advice = np.ones(n) * 50.0
    G = generate_barabasi_albert_graph(n=n, m=2, seed=42)
    W = get_degroot_trust_matrix(G, self_weight=0.5)

    # Pure social (alpha = 0)
    out_social = execute_hybrid_degroot_step(beliefs, W, ai_advice, alpha=0.0)
    np.testing.assert_allclose(out_social, 10.0, atol=1e-12)

    # Pure AI (alpha = 1)
    out_ai = execute_hybrid_degroot_step(beliefs, W, ai_advice, alpha=1.0)
    np.testing.assert_allclose(out_ai, 50.0, atol=1e-12)
    print("✓ Hybrid DeGroot boundary conditions (alpha=0.0 and alpha=1.0) verified.")


if __name__ == "__main__":
    test_page_diversity_theorem()
    test_degroot_matrix_stochasticity()
    test_hybrid_degroot_boundary_conditions()
    print("\nALL MATHEMATICAL INTEGRITY CHECKS PASSED SUCCESSFULLY.")

