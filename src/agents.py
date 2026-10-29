"""
Agents and Dynamics Module
Implements belief initialization, correlated AI advice generation (Kleinberg model),
and vectorized DeGroot belief updating.
"""

from typing import Tuple
import numpy as np


def initialize_human_beliefs(
    n: int, truth: float, noise_std: float = 15.0, seed: int = None
) -> np.ndarray:
    """
    Initializes human agent prior beliefs centered around the truth with independent noise.

    Formula:
        s_i(0) = truth + epsilon_i,  where epsilon_i ~ N(0, noise_std^2)
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    return truth + noise


def generate_correlated_ai_advice(
    n: int,
    truth: float,
    ai_bias: float = 2.0,
    ai_std: float = 5.0,
    correlation_rho: float = 0.85,
    seed: int = None,
) -> np.ndarray:
    """
    Generates AI advice according to Kleinberg & Raghavan's correlated noise model.

    Mathematical Formulation:
        Advice_i = truth + ai_bias + sqrt(rho) * eta_shared + sqrt(1 - rho) * eta_individual_i

    Where:
        - ai_bias (mu_AI): Systematic directional shift of the foundation model.
        - correlation_rho (rho in [0, 1]): Monoculture correlation.
            * rho = 1.0: Pure algorithmic monoculture (all agents get identical output).
            * rho = 0.0: Fully diverse independent AI assistance.
        - eta_shared ~ N(0, ai_std^2)
        - eta_individual_i ~ N(0, ai_std^2)

    Returns:
        1D numpy array of size n containing AI advice for each agent.
    """
    rng = np.random.default_rng(seed)
    rho = float(np.clip(correlation_rho, 0.0, 1.0))

    # Shared common-mode foundation model error
    eta_shared = rng.normal(loc=0.0, scale=ai_std)

    # Independent stochasticity across prompt variations/temperatures
    eta_individual = rng.normal(loc=0.0, scale=ai_std, size=n)

    # Correlated composite noise
    advice = (
        truth
        + ai_bias
        + np.sqrt(rho) * eta_shared
        + np.sqrt(1.0 - rho) * eta_individual
    )
    return advice


def execute_hybrid_degroot_step(
    beliefs: np.ndarray,
    W: np.ndarray,
    ai_advice: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Executes a single discrete time-step of the Hybrid DeGroot Consensus process.

    Mathematical Formulation:
        x(t+1) = (1 - alpha) * (W @ x(t)) + alpha * ai_advice

    Where:
        - beliefs: 1D array of current beliefs x(t).
        - W: Row-stochastic n x n trust matrix.
        - ai_advice: 1D array of algorithmic advice vector a(t).
        - alpha: Algorithmic Reliance Parameter in [0.0, 1.0].
            * alpha = 0.0: Pure peer social deliberation.
            * alpha = 1.0: Full reliance on AI model.

    Returns:
        1D array of updated beliefs x(t+1).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    social_deliberation = W @ beliefs
    updated_beliefs = (1.0 - alpha) * social_deliberation + alpha * ai_advice
    return updated_beliefs

