"""
Mathematical Metrics Module
Implements Page's Diversity Prediction Theorem, Shannon Entropy, and Monoculture Metrics.
"""

from typing import Dict, Tuple
import numpy as np


def compute_page_decomposition(estimates: np.ndarray, truth: float) -> Dict[str, float]:
    """
    Computes Scott Page's Diversity Prediction Theorem decomposition:
        Collective Error = Average Individual Error - Cognitive Diversity

    Mathematical Formulation:
        (c - theta)^2 = (1/n) * sum((s_i - theta)^2) - (1/n) * sum((s_i - c)^2)

    Args:
        estimates: 1D numpy array of agent predictions s_i.
        truth: Ground truth scalar value theta.

    Returns:
        Dictionary containing:
            - 'crowd_prediction': Mean estimate c
            - 'collective_error': (c - theta)^2
            - 'avg_individual_error': (1/n) * sum((s_i - theta)^2)
            - 'cognitive_diversity': (1/n) * sum((s_i - c)^2) (Sample variance)
            - 'identity_residual': Numerical check of the theorem equality (should be ~0.0)
    """
    s = np.asarray(estimates, dtype=np.float64)
    n = len(s)
    if n == 0:
        raise ValueError("Estimates array cannot be empty.")

    c = np.mean(s)
    collective_error = (c - truth) ** 2
    avg_individual_error = np.mean((s - truth) ** 2)
    cognitive_diversity = np.mean((s - c) ** 2)

    # Mathematical identity verification
    expected_collective = avg_individual_error - cognitive_diversity
    residual = abs(collective_error - expected_collective)

    return {
        "crowd_prediction": float(c),
        "collective_error": float(collective_error),
        "avg_individual_error": float(avg_individual_error),
        "cognitive_diversity": float(cognitive_diversity),
        "identity_residual": float(residual),
    }


def compute_shannon_entropy(
    estimates: np.ndarray, bins: int = 20, value_range: Tuple[float, float] = None
) -> float:
    """
    Computes Shannon Entropy of the belief distribution.

    Mathematical Formulation:
        H(S) = - sum(p_k * log2(p_k)) for p_k > 0

    Args:
        estimates: 1D numpy array of agent predictions.
        bins: Number of discrete histogram bins.
        value_range: Optional tuple (min, max) for bin bounds.

    Returns:
        Shannon Entropy in bits.
    """
    s = np.asarray(estimates, dtype=np.float64)
    if len(s) == 0:
        return 0.0

    counts, _ = np.histogram(s, bins=bins, range=value_range)
    total = np.sum(counts)
    if total == 0:
        return 0.0

    probs = counts / total
    non_zero = probs[probs > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return float(entropy)


def compute_monoculture_index(entropy: float, max_entropy: float) -> float:
    """
    Computes a normalized Monoculture Index in [0.0, 1.0]:
        M = 1.0 - (Entropy / Max_Entropy)

    A value of 0.0 represents maximum epistemic diversity.
    A value of 1.0 represents complete algorithmic monoculture (zero entropy).
    """
    if max_entropy <= 0.0:
        return 0.0
    val = 1.0 - (entropy / max_entropy)
    return float(np.clip(val, 0.0, 1.0))
