"""
Simulation Engine Module
Orchestrates discrete-event network belief evolution under hybrid peer-algorithmic consensus.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import networkx as nx

from metrics import compute_page_decomposition, compute_shannon_entropy, compute_monoculture_index
from networks import (
    generate_watts_strogatz_graph,
    generate_barabasi_albert_graph,
    get_degroot_trust_matrix,
    compute_network_metrics,
)
from agents import (
    initialize_human_beliefs,
    generate_correlated_ai_advice,
    execute_hybrid_degroot_step,
)


class SimulationEngine:
    """
    Simulation Engine modeling epistemic dynamics across complex networks
    with varying reliance on a centralized, correlated foundation model.
    """

    def __init__(
        self,
        num_agents: int = 250,
        topology: str = "watts_strogatz",
        truth: float = 100.0,
        human_noise_std: float = 15.0,
        ai_bias: float = 2.0,
        ai_std: float = 4.0,
        correlation_rho: float = 0.85,
        alpha: float = 0.3,
        timesteps: int = 30,
        self_weight: float = 0.5,
        seed: Optional[int] = None,
        # Topology-specific kwargs
        ws_k: int = 6,
        ws_p: float = 0.1,
        ba_m: int = 3,
        er_p: float = 0.05,
    ):
        self.num_agents = num_agents
        self.topology = topology.lower()
        self.truth = truth
        self.human_noise_std = human_noise_std
        self.ai_bias = ai_bias
        self.ai_std = ai_std
        self.correlation_rho = correlation_rho
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.timesteps = timesteps
        self.self_weight = self_weight
        self.seed = seed

        self.ws_k = ws_k
        self.ws_p = ws_p
        self.ba_m = ba_m
        self.er_p = er_p

        # Initialize network graph
        self.graph = self._build_graph()
        self.actual_agents = self.graph.number_of_nodes()
        self.W = get_degroot_trust_matrix(self.graph, self_weight=self.self_weight)
        self.network_metrics = compute_network_metrics(self.graph)

    def _build_graph(self) -> nx.Graph:
        """Instantiates graph according to requested topology."""
        if self.topology == "watts_strogatz":
            return generate_watts_strogatz_graph(
                n=self.num_agents, k=self.ws_k, p=self.ws_p, seed=self.seed
            )
        elif self.topology == "barabasi_albert":
            return generate_barabasi_albert_graph(
                n=self.num_agents, m=self.ba_m, seed=self.seed
            )
        elif self.topology == "erdos_renyi":
            G = nx.erdos_renyi_graph(n=self.num_agents, p=self.er_p, seed=self.seed)
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            return G
        else:
            raise ValueError(f"Unknown network topology: {self.topology}")

    def run(self) -> Dict[str, Any]:
        """
        Executes the discrete-event simulation across T timesteps.

        Returns:
            Dictionary containing:
                - 'trajectory': List of step-by-step metrics
                - 'final_metrics': Summary metrics at t = T
                - 'network_metrics': Structural graph properties
                - 'parameters': Run configuration
        """
        # Step 0: Initialize human agent beliefs
        beliefs = initialize_human_beliefs(
            n=self.actual_agents,
            truth=self.truth,
            noise_std=self.human_noise_std,
            seed=self.seed,
        )

        trajectory: List[Dict[str, float]] = []

        # Baseline entropy for normalization (maximum entropy across initial distribution)
        baseline_entropy = np.log2(20.0)  # Reference scale for 20 bins

        for t in range(self.timesteps + 1):
            # Compute Page decomposition & information-theoretic metrics at current state
            decomp = compute_page_decomposition(beliefs, self.truth)
            entropy = compute_shannon_entropy(beliefs, bins=20)
            monoculture_idx = compute_monoculture_index(entropy, baseline_entropy)

            step_record = {
                "t": t,
                "collective_error": decomp["collective_error"],
                "avg_individual_error": decomp["avg_individual_error"],
                "cognitive_diversity": decomp["cognitive_diversity"],
                "shannon_entropy": entropy,
                "monoculture_index": monoculture_idx,
                "mean_belief": decomp["crowd_prediction"],
                "std_belief": float(np.std(beliefs)),
            }
            trajectory.append(step_record)

            if t < self.timesteps:
                # Generate step-specific correlated AI advice
                step_seed = (self.seed + t * 997) if self.seed is not None else None
                ai_advice = generate_correlated_ai_advice(
                    n=self.actual_agents,
                    truth=self.truth,
                    ai_bias=self.ai_bias,
                    ai_std=self.ai_std,
                    correlation_rho=self.correlation_rho,
                    seed=step_seed,
                )

                # Update beliefs via hybrid DeGroot consensus
                beliefs = execute_hybrid_degroot_step(
                    beliefs=beliefs,
                    W=self.W,
                    ai_advice=ai_advice,
                    alpha=self.alpha,
                )

        final_metrics = trajectory[-1]

        return {
            "trajectory": trajectory,
            "final_metrics": final_metrics,
            "network_metrics": self.network_metrics,
            "parameters": {
                "num_agents": self.actual_agents,
                "topology": self.topology,
                "alpha": self.alpha,
                "correlation_rho": self.correlation_rho,
                "ai_bias": self.ai_bias,
                "ai_std": self.ai_std,
                "human_noise_std": self.human_noise_std,
                "timesteps": self.timesteps,
                "seed": self.seed,
            },
        }

