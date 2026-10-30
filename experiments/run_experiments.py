"""
Experiment Runner and Publication Visualization Generator (Macro-Scale Suite).
Executes systematic parameter sweeps across macro-scale networks (N = 1,000 agents),
evaluating 3 network topologies, 11 reliance levels (alpha), and 3 correlation regimes (rho),
saving raw data (2,970 total runs) and generating high-DPI publication figures.
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root and src are available on path
ROOT_PATH = str(Path(__file__).resolve().parent.parent)
SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
for p in (ROOT_PATH, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.simulation import SimulationEngine
from src.networks import (
    generate_watts_strogatz_graph,
    generate_barabasi_albert_graph,
    get_degroot_trust_matrix,
    compute_network_metrics,
)
import networkx as nx

# Ensure output directories exist
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def run_parameter_sweep() -> pd.DataFrame:
    """
    Executes systematic macro-scale sweep across:
        - Topologies: Watts-Strogatz (Small-World), Barabási-Albert (Scale-Free), Erdős-Rényi (Random)
        - Reliance (alpha): 0.0 to 1.0 (step 0.1)
        - Monoculture regimes (rho):
            * 0.90 (Monopoly Monoculture - single dominant foundation model)
            * 0.50 (Oligopoly - multi-model ecosystem, e.g. OpenAI + Anthropic + Google)
            * 0.00 (Polyculture Control - independent, unconstrained models)
        - Agent Population: N = 1,000 agents per network
        - Monte Carlo trials: 30 independent runs per condition (2,970 total runs)
    """
    topologies = ["watts_strogatz", "barabasi_albert", "erdos_renyi"]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rhos = [0.90, 0.50, 0.00]
    num_trials = 30
    num_agents = 1000
    timesteps = 30

    records: List[Dict[str, Any]] = []
    total_runs = len(topologies) * len(alphas) * len(rhos) * num_trials
    print(f"[*] Commencing Macro-Scale Monte Carlo sweep: {total_runs} total runs (N = {num_agents} agents)...")
    start_time = time.time()

    run_idx = 0
    for topology in topologies:
        print(f"[*] Pre-computing network topology and trust matrix for {topology} (N={num_agents})...")
        if topology == "watts_strogatz":
            cached_graph = generate_watts_strogatz_graph(n=num_agents, k=6, p=0.1, seed=42)
        elif topology == "barabasi_albert":
            cached_graph = generate_barabasi_albert_graph(n=num_agents, m=3, seed=42)
        elif topology == "erdos_renyi":
            cached_graph = nx.erdos_renyi_graph(n=num_agents, p=0.05, seed=42)
            if not nx.is_connected(cached_graph):
                largest_cc = max(nx.connected_components(cached_graph), key=len)
                cached_graph = cached_graph.subgraph(largest_cc).copy()
        
        cached_W = get_degroot_trust_matrix(cached_graph, self_weight=0.5)
        cached_metrics = compute_network_metrics(cached_graph)
        print(f"    [✓] {topology} ready: Clustering={cached_metrics['avg_clustering_coeff']:.3f}, PathLength={cached_metrics['avg_path_length']:.3f}")

        for rho in rhos:
            for alpha in alphas:
                for trial in range(num_trials):
                    seed = 20000 + trial * 47 + int(alpha * 100) + int(rho * 1000)
                    sim = SimulationEngine(
                        num_agents=num_agents,
                        topology=topology,
                        alpha=alpha,
                        correlation_rho=rho,
                        timesteps=timesteps,
                        seed=seed,
                        graph=cached_graph,
                        W=cached_W,
                        network_metrics=cached_metrics,
                    )
                    res = sim.run()
                    final = res["final_metrics"]

                    records.append({
                        "topology": topology,
                        "alpha": alpha,
                        "rho": rho,
                        "trial": trial,
                        "num_agents": num_agents,
                        "collective_error": final["collective_error"],
                        "avg_individual_error": final["avg_individual_error"],
                        "cognitive_diversity": final["cognitive_diversity"],
                        "shannon_entropy": final["shannon_entropy"],
                        "monoculture_index": final["monoculture_index"],
                        "mean_belief": final["mean_belief"],
                        "std_belief": final["std_belief"],
                        "avg_clustering_coeff": res["network_metrics"]["avg_clustering_coeff"],
                        "avg_path_length": res["network_metrics"]["avg_path_length"],
                    })
                    run_idx += 1

                if (run_idx % 300 == 0) or (run_idx == total_runs):
                    progress = (run_idx / total_runs) * 100
                    elapsed = time.time() - start_time
                    print(f"    -> Progress: {run_idx}/{total_runs} runs ({progress:.1f}%) in {elapsed:.1f}s...")

    elapsed = time.time() - start_time
    print(f"[✓] Macro-Scale Sweep completed in {elapsed:.2f} seconds ({elapsed/total_runs*1000:.2f} ms/run).")

    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "simulation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] Raw experimental data saved to: {csv_path}")
    return df


def generate_publication_figures(df: pd.DataFrame):
    """Generates high-DPI, publication-grade academic figures (PNG and PDF)."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })

    topo_labels = {
        "watts_strogatz": "Watts-Strogatz (Small-World)",
        "barabasi_albert": "Barabási-Albert (Scale-Free)",
        "erdos_renyi": "Erdős-Rényi (Random)",
    }
    topo_colors = {
        "watts_strogatz": "#1f77b4",  # Blue
        "barabasi_albert": "#d62728",  # Red
        "erdos_renyi": "#2ca02c",      # Green
    }

    # =========================================================================
    # Figure 1: 3-Panel Phase Transition Across Correlation Regimes
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    regimes = [
        (0.90, "A: Monopoly Monoculture (ρ = 0.90)"),
        (0.50, "B: Oligopoly / Multi-Model (ρ = 0.50)"),
        (0.00, "C: Polyculture Control (ρ = 0.00)"),
    ]

    for idx, (rho, title) in enumerate(regimes):
        ax = axes[idx]
        sub_df = df[df["rho"] == rho]

        for topo, group in sub_df.groupby("topology"):
            summary = group.groupby("alpha")["collective_error"].agg(["mean", "sem"]).reset_index()
            ax.plot(
                summary["alpha"], summary["mean"],
                label=topo_labels[topo],
                color=topo_colors[topo],
                marker="o" if topo == "watts_strogatz" else ("s" if topo == "barabasi_albert" else "^"),
            )
            ax.fill_between(
                summary["alpha"],
                summary["mean"] - 1.96 * summary["sem"],
                summary["mean"] + 1.96 * summary["sem"],
                color=topo_colors[topo],
                alpha=0.15,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Algorithmic Reliance Parameter (α)")
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.set_ylabel(r"Collective Squared Error $(c - \theta)^2$")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.suptitle("Figure 1: Collective Error Phase Transitions Across Macro-Scale Networks (N = 1,000)", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_collective_error_phase_transition.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_collective_error_phase_transition.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 1: 3-Panel Phase Transition")

    # =========================================================================
    # Figure 2: Cognitive Diversity Collapse vs. Reliance (alpha)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for idx, (rho, title) in enumerate([(0.90, "A: High Monoculture (ρ = 0.90)"), (0.50, "B: Oligopoly (ρ = 0.50)")]):
        ax = axes[idx]
        sub_df = df[df["rho"] == rho]

        for topo, group in sub_df.groupby("topology"):
            summary = group.groupby("alpha")["cognitive_diversity"].agg(["mean", "sem"]).reset_index()
            ax.plot(
                summary["alpha"], summary["mean"],
                label=topo_labels[topo],
                color=topo_colors[topo],
                marker="o" if topo == "watts_strogatz" else ("s" if topo == "barabasi_albert" else "^"),
            )
            ax.fill_between(
                summary["alpha"],
                summary["mean"] - 1.96 * summary["sem"],
                summary["mean"] + 1.96 * summary["sem"],
                color=topo_colors[topo],
                alpha=0.15,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Algorithmic Reliance Parameter (α)")
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.set_ylabel(r"Cognitive Diversity $\frac{1}{n} \sum (s_i - c)^2$ (Variance)")
            ax.legend(frameon=True, facecolor="white")

    plt.suptitle("Figure 2: Cognitive Diversity Collapse Under Centralized Foundation Models (N = 1,000)", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cognitive_diversity_collapse.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_cognitive_diversity_collapse.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 2: Cognitive Diversity Collapse")

    # =========================================================================
    # Figure 3: Monoculture Index & Epistemic Homogenization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.2))
    mono_df = df[df["rho"] == 0.90]

    for topo, group in mono_df.groupby("topology"):
        summary = group.groupby("alpha")["monoculture_index"].agg(["mean", "sem"]).reset_index()
        ax.plot(
            summary["alpha"], summary["mean"],
            label=topo_labels[topo],
            color=topo_colors[topo],
            marker="o" if topo == "watts_strogatz" else ("s" if topo == "barabasi_albert" else "^"),
        )
        ax.fill_between(
            summary["alpha"],
            summary["mean"] - 1.96 * summary["sem"],
            summary["mean"] + 1.96 * summary["sem"],
            color=topo_colors[topo],
            alpha=0.15,
        )

    ax.set_title("Figure 3: Monoculture Index: Epistemic Homogenization Across Topologies (N = 1,000)", fontweight="bold")
    ax.set_xlabel("Algorithmic Reliance Parameter (α)")
    ax.set_ylabel(r"Monoculture Index $\mathcal{M} = 1 - (H / H_{\max})$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_monoculture_index.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_monoculture_index.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 3: Monoculture Index")

    # =========================================================================
    # Figure 4: Page's Theorem Three-Component Decomposition
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ws_mono = mono_df[mono_df["topology"] == "watts_strogatz"]
    ws_summary = ws_mono.groupby("alpha")[["collective_error", "avg_individual_error", "cognitive_diversity"]].mean().reset_index()

    ax.plot(ws_summary["alpha"], ws_summary["avg_individual_error"], label=r"Avg Individual Error: $\overline{(s_i - \theta)^2}$", color="#e67e22", linestyle="--", marker="d")
    ax.plot(ws_summary["alpha"], ws_summary["cognitive_diversity"], label=r"Cognitive Diversity: $\overline{(s_i - c)^2}$", color="#27ae60", linestyle="-.", marker="^")
    ax.plot(ws_summary["alpha"], ws_summary["collective_error"], label=r"Collective Error: $(c - \theta)^2$", color="#8e44ad", linestyle="-", marker="o", linewidth=2.8)

    ax.set_title("Figure 4: Empirical Verification of Page's Diversity Theorem (Watts-Strogatz, N = 1,000)", fontweight="bold")
    ax.set_xlabel("Algorithmic Reliance Parameter (α)")
    ax.set_ylabel("Error / Diversity Magnitude")
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.text(
        0.05, 0.90,
        r"$\mathbf{Collective\ Error} = \mathbf{Individual\ Error} - \mathbf{Diversity}$",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da")
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_page_decomposition_verification.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_page_decomposition_verification.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 4: Page's Theorem Decomposition")


def generate_time_series_trajectory():
    """Generates Figure 5: Step-by-step belief trajectories over timesteps t for different alpha at N=1,000."""
    alphas_to_test = [0.0, 0.2, 0.5, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas_to_test):
        ax = axes[idx]
        sim = SimulationEngine(
            num_agents=1000,
            topology="watts_strogatz",
            alpha=alpha,
            correlation_rho=0.90,
            timesteps=25,
            seed=42,
        )
        res = sim.run()
        traj = pd.DataFrame(res["trajectory"])

        ax.plot(traj["t"], traj["mean_belief"], color="#2c3e50", label=r"Crowd Mean $c(t)$", linewidth=2.5)
        ax.fill_between(
            traj["t"],
            traj["mean_belief"] - traj["std_belief"],
            traj["mean_belief"] + traj["std_belief"],
            color="#3498db",
            alpha=0.25,
            label="Belief Spread (±1 SD)",
        )
        ax.axhline(100.0, color="#e74c3c", linestyle=":", linewidth=1.8, label=r"Ground Truth ($\theta=100$)")

        ax.set_title(f"α = {alpha:.1f} ({'Pure Social' if alpha==0 else ('Dominated AI' if alpha>=0.8 else 'Hybrid')})", fontweight="bold")
        ax.set_xlabel("Simulation Timestep (t)")
        ax.set_ylabel("Agent Belief Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white")

    plt.suptitle("Figure 5: Network Deliberation Trajectories and Belief Convergence (Watts-Strogatz, N = 1,000)", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_trajectory_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_trajectory_convergence.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 5: Network Deliberation Trajectories")


if __name__ == "__main__":
    df = run_parameter_sweep()
    generate_publication_figures(df)
    generate_time_series_trajectory()
    print("\nALL MACRO-SCALE EXPERIMENTS AND PUBLICATION FIGURES SUCCESSFULLY GENERATED.")
