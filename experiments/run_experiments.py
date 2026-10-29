"""
Experiment Runner and Publication Visualization Generator.
Executes systematic parameter sweeps across network topologies, algorithmic reliance (alpha),
and correlation (rho), saving raw data and generating high-DPI publication figures.
"""

import sys
from pathlib import Path
import time
import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from simulation import SimulationEngine

# Ensure output directories exist
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def run_parameter_sweep() -> pd.DataFrame:
    """
    Executes systematic sweep across:
        - Topologies: Watts-Strogatz (Small-World), Barabási-Albert (Scale-Free), Erdős-Rényi (Random)
        - Reliance (alpha): 0.0 to 1.0 (step 0.1)
        - Monoculture correlation (rho): 0.85 (monoculture) vs. 0.0 (independent/polyculture)
        - Monte Carlo trials: 15 independent runs per parameter combination
    """
    topologies = ["watts_strogatz", "barabasi_albert", "erdos_renyi"]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rhos = [0.85, 0.0]
    num_trials = 15
    num_agents = 250
    timesteps = 30

    records: List[Dict[str, Any]] = []
    total_runs = len(topologies) * len(alphas) * len(rhos) * num_trials
    print(f"[*] Commencing Monte Carlo sweep: {total_runs} total simulation runs...")
    start_time = time.time()

    run_idx = 0
    for topology in topologies:
        for rho in rhos:
            for alpha in alphas:
                for trial in range(num_trials):
                    seed = 10000 + trial * 31 + int(alpha * 100)
                    sim = SimulationEngine(
                        num_agents=num_agents,
                        topology=topology,
                        alpha=alpha,
                        correlation_rho=rho,
                        timesteps=timesteps,
                        seed=seed,
                    )
                    res = sim.run()
                    final = res["final_metrics"]

                    records.append({
                        "topology": topology,
                        "alpha": alpha,
                        "rho": rho,
                        "trial": trial,
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

    elapsed = time.time() - start_time
    print(f"[✓] Sweep completed in {elapsed:.2f} seconds ({elapsed/total_runs*1000:.2f} ms/run).")

    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "simulation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] Raw experimental data saved to: {csv_path}")
    return df


def generate_publication_figures(df: pd.DataFrame):
    """Generates high-DPI, publication-grade academic figures (PNG and PDF)."""
    # Publication aesthetic styling
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
    # Figure 1: Phase Transition - Collective Error vs. Reliance (alpha)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for idx, (rho, title) in enumerate([(0.85, "A: Algorithmic Monoculture (ρ = 0.85)"),
                                        (0.0, "B: Independent AI Advice / Polyculture (ρ = 0.0)")]):
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
            ax.set_ylabel("Collective Squared Error $(c - θ)^2$")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.suptitle("Figure 1: Collective Error Phase Transition Across Network Topologies", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_collective_error_phase_transition.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_collective_error_phase_transition.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 1: Collective Error Phase Transition")

    # =========================================================================
    # Figure 2: Cognitive Diversity Collapse vs. Reliance (alpha)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mono_df = df[df["rho"] == 0.85]

    for topo, group in mono_df.groupby("topology"):
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

    ax.set_title("Figure 2: Cognitive Diversity Collapse Under Algorithmic Monoculture (ρ = 0.85)", fontweight="bold")
    ax.set_xlabel("Algorithmic Reliance Parameter (α)")
    ax.set_ylabel(r"Cognitive Diversity: $\frac{1}{n} \sum (s_i - c)^2$ (Variance)")
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cognitive_diversity_collapse.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_cognitive_diversity_collapse.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 2: Cognitive Diversity Collapse")

    # =========================================================================
    # Figure 3: Monoculture Index & Shannon Entropy
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 5))

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

    ax.set_title("Figure 3: Monoculture Index: Epistemic Homogenization Across Topologies", fontweight="bold")
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
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ws_mono = mono_df[mono_df["topology"] == "watts_strogatz"]
    ws_summary = ws_mono.groupby("alpha")[["collective_error", "avg_individual_error", "cognitive_diversity"]].mean().reset_index()

    ax.plot(ws_summary["alpha"], ws_summary["avg_individual_error"], label="Avg Individual Error: $\overline{(s_i - θ)^2}$", color="#e67e22", linestyle="--", marker="d")
    ax.plot(ws_summary["alpha"], ws_summary["cognitive_diversity"], label="Cognitive Diversity: $\overline{(s_i - c)^2}$", color="#27ae60", linestyle="-.", marker="^")
    ax.plot(ws_summary["alpha"], ws_summary["collective_error"], label="Collective Error: $(c - θ)^2$", color="#8e44ad", linestyle="-", marker="o", linewidth=2.8)

    ax.set_title("Figure 4: Empirical Verification of Page's Diversity Decomposition (Watts-Strogatz)", fontweight="bold")
    ax.set_xlabel("Algorithmic Reliance Parameter (α)")
    ax.set_ylabel("Error / Diversity Magnitude")
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotate the theorem identity on the plot
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
    """Generates Figure 5: Step-by-step belief trajectories over timesteps t for different alpha."""
    alphas_to_test = [0.0, 0.2, 0.5, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas_to_test):
        ax = axes[idx]
        sim = SimulationEngine(
            num_agents=150,
            topology="watts_strogatz",
            alpha=alpha,
            correlation_rho=0.85,
            timesteps=25,
            seed=42,
        )
        res = sim.run()
        traj = pd.DataFrame(res["trajectory"])

        ax.plot(traj["t"], traj["mean_belief"], color="#2c3e50", label="Crowd Mean $c(t)$", linewidth=2.5)
        ax.fill_between(
            traj["t"],
            traj["mean_belief"] - traj["std_belief"],
            traj["mean_belief"] + traj["std_belief"],
            color="#3498db",
            alpha=0.25,
            label="Belief Spread (±1 SD)",
        )
        ax.axhline(100.0, color="#e74c3c", linestyle=":", linewidth=1.8, label="Ground Truth ($θ=100$)")

        ax.set_title(f"α = {alpha:.1f} ({'Pure Social' if alpha==0 else ('Dominated AI' if alpha>=0.8 else 'Hybrid')})", fontweight="bold")
        ax.set_xlabel("Simulation Timestep (t)")
        ax.set_ylabel("Agent Belief Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white")

    plt.suptitle("Figure 5: Network Deliberation Trajectories and Belief Convergence (Watts-Strogatz)", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_trajectory_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_trajectory_convergence.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Saved Figure 5: Network Deliberation Trajectories")


if __name__ == "__main__":
    df = run_parameter_sweep()
    generate_publication_figures(df)
    generate_time_series_trajectory()
    print("\nALL SIMULATION EXPERIMENTS AND PUBLICATION FIGURES SUCCESSFULLY GENERATED.")

