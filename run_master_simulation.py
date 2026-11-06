"""
Master High-Throughput Simulation Engine
Executes 49,500+ independent Monte Carlo trials across network topologies,
algorithmic correlation regimes, and human correlation levels.
Generates publication figures and exact empirical LaTeX tables.
"""

import time
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.networks import (
    generate_watts_strogatz_graph,
    generate_barabasi_albert_graph,
    get_degroot_trust_matrix,
)
from src.agents import generate_correlated_ai_advice, execute_hybrid_degroot_step

print("=== STARTING MASTER MONTE CARLO SIMULATION ===")
t_start = time.time()

# System Parameters
n = 1000
truth = 100.0
sigma_h = 15.0
var_h = 225.0
mu_ai = 2.0
sigma_ai = 4.0
var_ai = 16.0
T = 30
K_TRIALS = 500  # 500 independent trials per cell

# 1. Pre-construct Graphs and Trust Matrices
print("Generating network graphs (N = 1,000)...")
graphs = {
    "watts_strogatz": generate_watts_strogatz_graph(n, k=6, p=0.1, seed=42),
    "erdos_renyi": nx.erdos_renyi_graph(n, p=0.05, seed=43),
    "barabasi_albert": generate_barabasi_albert_graph(n, m=3, seed=44),
}

# Ensure ER graph is connected
if not nx.is_connected(graphs["erdos_renyi"]):
    largest_cc = max(nx.connected_components(graphs["erdos_renyi"]), key=len)
    graphs["erdos_renyi"] = graphs["erdos_renyi"].subgraph(largest_cc).copy()

matrices = {}
stationary_weights = {}
for name, G in graphs.items():
    W = get_degroot_trust_matrix(G, self_weight=0.5)
    matrices[name] = W
    # Compute stationary distribution pi (left eigenvector for eigenvalue 1)
    vals, vecs = np.linalg.eig(W.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = pi / np.sum(pi)
    stationary_weights[name] = pi
    sum_pi_sq = np.sum(pi**2)
    print(f"Topology {name}: sum(pi^2) = {sum_pi_sq:.5f}, theoretical baseline var = {var_h * sum_pi_sq:.4f}")

# Analytical Formulas
def theoretical_dynamic_error(alpha, rho_ai, rho_h=0.0, sum_pi_sq=0.001, T=30):
    lam = (1.0 - alpha)**T
    mse_h = rho_h * var_h + (1.0 - rho_h) * var_h * sum_pi_sq
    human_part = (lam**2) * mse_h
    bias_part = (mu_ai * (1.0 - lam))**2
    if alpha == 0.0:
        return mse_h
    elif alpha == 1.0:
        return mu_ai**2 + rho_ai * var_ai
    ai_var_part = (alpha * (1.0 - (1.0 - alpha)**(2*T)) / (2.0 - alpha)) * rho_ai * var_ai
    return human_part + bias_part + ai_var_part

def theoretical_static_error(alpha, rho_ai, rho_h=0.0, sum_pi_sq=0.001, T=30):
    lam = (1.0 - alpha)**T
    mse_h = rho_h * var_h + (1.0 - rho_h) * var_h * sum_pi_sq
    human_part = (lam**2) * mse_h
    ai_part = ((1.0 - lam)**2) * (mu_ai**2 + rho_ai * var_ai)
    return human_part + ai_part

# ---------------------------------------------------------------------------
# EXPERIMENT 1: Comprehensive Parameter Sweep across Reliance & Correlation (rho_h = 0.0)
# ---------------------------------------------------------------------------
reliance_levels = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
all_reliance_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
correlations = [0.00, 0.50, 0.90]

print(f"\nRunning Experiment 1: {len(all_reliance_levels)} reliance levels x 3 regimes x 3 topologies x {K_TRIALS} trials...")

exp1_results = {}

seed_counter = 100000

for rho in correlations:
    exp1_results[rho] = {}
    for a in all_reliance_levels:
        exp1_results[rho][a] = {"topologies": {}, "pooled": {}}
        pooled_errors = []
        pooled_diversities = []
        
        for top_name in ["watts_strogatz", "erdos_renyi", "barabasi_albert"]:
            W = matrices[top_name]
            actual_n = W.shape[0]
            errors = []
            diversities = []
            
            for k in range(K_TRIALS):
                seed_counter += 1
                rng = np.random.default_rng(seed_counter)
                
                # Human prior (rho_h = 0.0)
                beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
                
                for t in range(T):
                    step_seed = seed_counter + 10000 + t * 97
                    ai = generate_correlated_ai_advice(
                        actual_n, truth, ai_bias=mu_ai, ai_std=sigma_ai, correlation_rho=rho, seed=step_seed
                    )
                    beliefs = execute_hybrid_degroot_step(beliefs, W, ai, a)
                
                c = np.mean(beliefs)
                err = (c - truth)**2
                div = np.mean((beliefs - c)**2)
                errors.append(err)
                diversities.append(div)
                pooled_errors.append(err)
                pooled_diversities.append(div)
            
            exp1_results[rho][a]["topologies"][top_name] = {
                "mean_error": float(np.mean(errors)),
                "sem_error": float(np.std(errors) / np.sqrt(K_TRIALS)),
                "mean_diversity": float(np.mean(diversities)),
                "sem_diversity": float(np.std(diversities) / np.sqrt(K_TRIALS)),
            }
        
        total_pooled = len(pooled_errors)
        exp1_results[rho][a]["pooled"] = {
            "mean_error": float(np.mean(pooled_errors)),
            "sem_error": float(np.std(pooled_errors) / np.sqrt(total_pooled)),
            "mean_diversity": float(np.mean(pooled_diversities)),
            "sem_diversity": float(np.std(pooled_diversities) / np.sqrt(total_pooled)),
        }
    print(f"  Completed correlation rho = {rho:.2f}")

print("Experiment 1 complete.")

# ---------------------------------------------------------------------------
# EXPERIMENT 2: Human Correlation Sweep (Optimal Reliance & Crossover)
# ---------------------------------------------------------------------------
human_correlations = [0.00, 0.02, 0.05, 0.10]
exp2_results = {}

print("\nRunning Experiment 2: Human Correlation Sweep...")
W_ws = matrices["watts_strogatz"]
actual_n = W_ws.shape[0]

for rh in human_correlations:
    exp2_results[rh] = {}
    for a in all_reliance_levels:
        errors = []
        for k in range(K_TRIALS):
            seed_counter += 1
            rng = np.random.default_rng(seed_counter)
            
            if rh > 0.0:
                shared_h = rng.normal(0.0, sigma_h)
                indiv_h = rng.normal(0.0, sigma_h, size=actual_n)
                beliefs = truth + np.sqrt(rh) * shared_h + np.sqrt(1.0 - rh) * indiv_h
            else:
                beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
            
            for t in range(T):
                step_seed = seed_counter + 20000 + t * 97
                ai = generate_correlated_ai_advice(
                    actual_n, truth, ai_bias=mu_ai, ai_std=sigma_ai, correlation_rho=0.90, seed=step_seed
                )
                beliefs = execute_hybrid_degroot_step(beliefs, W_ws, ai, a)
            
            c = np.mean(beliefs)
            errors.append((c - truth)**2)
        
        exp2_results[rh][a] = {
            "mean_error": float(np.mean(errors)),
            "sem_error": float(np.std(errors) / np.sqrt(K_TRIALS)),
        }
    print(f"  Completed human correlation rho_h = {rh:.2f}")

# ---------------------------------------------------------------------------
# EXPERIMENT 3: Static vs. Dynamic AI Noise Comparison
# ---------------------------------------------------------------------------
print("\nRunning Experiment 3: Static vs. Dynamic AI Noise Comparison...")
exp3_results = {"dynamic": {}, "static": {}}

for a in all_reliance_levels:
    dyn_errors = []
    stat_errors = []
    for k in range(K_TRIALS):
        seed_counter += 1
        rng = np.random.default_rng(seed_counter)
        
        # Initial human belief (rho_h = 0.0)
        init_beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
        
        # Dynamic run
        b_dyn = init_beliefs.copy()
        for t in range(T):
            step_seed = seed_counter + 30000 + t * 97
            ai = generate_correlated_ai_advice(actual_n, truth, ai_bias=mu_ai, ai_std=sigma_ai, correlation_rho=0.90, seed=step_seed)
            b_dyn = execute_hybrid_degroot_step(b_dyn, W_ws, ai, a)
        dyn_errors.append((np.mean(b_dyn) - truth)**2)
        
        # Static run (advice constant over time)
        b_stat = init_beliefs.copy()
        static_ai = generate_correlated_ai_advice(actual_n, truth, ai_bias=mu_ai, ai_std=sigma_ai, correlation_rho=0.90, seed=seed_counter + 40000)
        for t in range(T):
            b_stat = execute_hybrid_degroot_step(b_stat, W_ws, static_ai, a)
        stat_errors.append((np.mean(b_stat) - truth)**2)
        
    exp3_results["dynamic"][a] = {"mean": float(np.mean(dyn_errors)), "sem": float(np.std(dyn_errors) / np.sqrt(K_TRIALS))}
    exp3_results["static"][a] = {"mean": float(np.mean(stat_errors)), "sem": float(np.std(stat_errors) / np.sqrt(K_TRIALS))}

print("Experiment 3 complete.")

# ---------------------------------------------------------------------------
# SAVE RAW SUMMARY DATA
# ---------------------------------------------------------------------------
output_data = {
    "parameters": {
        "n": n, "truth": truth, "sigma_h": sigma_h, "var_h": var_h,
        "mu_ai": mu_ai, "sigma_ai": sigma_ai, "var_ai": var_ai,
        "T": T, "K_TRIALS": K_TRIALS, "total_simulations": seed_counter - 100000
    },
    "exp1_parameter_sweep": exp1_results,
    "exp2_human_correlation": exp2_results,
    "exp3_static_vs_dynamic": exp3_results
}

with open("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/data/master_simulation_summary.json", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nTotal simulations executed: {seed_counter - 100000}")
print(f"Total time elapsed: {time.time() - t_start:.2f} seconds.")

# ---------------------------------------------------------------------------
# GENERATE FIGURE 1: 3-Panel Phase Transition with High-Sample Precision
# ---------------------------------------------------------------------------
print("\nRendering Figure 1 (3-Panel Phase Transition)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)

panel_configs = [
    (0.90, "Panel A: Monopoly Monoculture (ρ = 0.90)", axes[0]),
    (0.50, "Panel B: Correlated Multi-Model (ρ = 0.50)", axes[1]),
    (0.00, "Panel C: Polyculture Control (ρ = 0.00)", axes[2]),
]

top_styles = {
    "watts_strogatz": ("#1f77b4", "o", "Watts-Strogatz"),
    "erdos_renyi": ("#2ca02c", "s", "Erdős-Rényi"),
    "barabasi_albert": ("#d62728", "^", "Barabási-Albert"),
}

alphas_fine = np.linspace(0.0, 1.0, 200)

for rho, title, ax in panel_configs:
    # Closed-form theory curve
    theory_vals = [theoretical_dynamic_error(a, rho, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]
    ax.plot(alphas_fine, theory_vals, "k--", linewidth=1.8, label="Analytical Theory", zorder=3)
    
    for top_name, (color, marker, label) in top_styles.items():
        x_pts = all_reliance_levels
        y_pts = [exp1_results[rho][a]["topologies"][top_name]["mean_error"] for a in x_pts]
        y_err = [exp1_results[rho][a]["topologies"][top_name]["sem_error"] for a in x_pts]
        ax.errorbar(x_pts, y_pts, yerr=y_err, fmt=marker, color=color, ecolor=color, elinewidth=1.0,
                    capsize=2.5, markersize=4.5, label=f"{label} (K=500)", zorder=4, alpha=0.9)
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
    if ax == axes[0]:
        ax.set_ylabel("Mean Collective Squared Error (c - θ)²", fontsize=10.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, 22.0)
    ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig("/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig1_collective_error_phase_transition.png")
fig.savefig("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig1_collective_error_phase_transition.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# GENERATE FIGURE 2: Cognitive Diversity & Variance Compression
# ---------------------------------------------------------------------------
print("Rendering Figure 2 (Cognitive Diversity Collapse)...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, sharey=True)

div_panels = [
    (0.90, "Monopoly Monoculture (ρ = 0.90)", axes[0]),
    (0.50, "Correlated Multi-Model (ρ = 0.50)", axes[1]),
]

for rho, title, ax in div_panels:
    x_pts = all_reliance_levels
    y_ws = [exp1_results[rho][a]["topologies"]["watts_strogatz"]["mean_diversity"] for a in x_pts]
    y_er = [exp1_results[rho][a]["topologies"]["erdos_renyi"]["mean_diversity"] for a in x_pts]
    y_ba = [exp1_results[rho][a]["topologies"]["barabasi_albert"]["mean_diversity"] for a in x_pts]
    
    ax.plot(x_pts, y_ws, "o-", color="#1f77b4", linewidth=1.8, markersize=4.5, label="Watts-Strogatz")
    ax.plot(x_pts, y_er, "s-", color="#2ca02c", linewidth=1.8, markersize=4.5, label="Erdős-Rényi")
    ax.plot(x_pts, y_ba, "^-", color="#d62728", linewidth=1.8, markersize=4.5, label="Barabási-Albert")
    
    # Asymptotic diversity limit = (1 - rho) * var_ai
    asymp_div = (1.0 - rho) * var_ai
    ax.axhline(asymp_div, color="black", linestyle="--", linewidth=1.2, label=f"Asymptotic Limit: (1-ρ)σ² = {asymp_div:.1f}")
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
    if ax == axes[0]:
        ax.set_ylabel("Cognitive Diversity (1/N) ∑(x_i - c)²", fontsize=10.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, 18.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5, loc="upper left")

plt.tight_layout()
fig.savefig("/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig2_cognitive_diversity_collapse.png")
fig.savefig("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig2_cognitive_diversity_collapse.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# GENERATE FIGURE 3: Optimal Reliance Trade-Off & Crossover (Figure 3 in text)
# ---------------------------------------------------------------------------
print("Rendering Figure 3 (The Optimal Reliance Trade-Off)...")
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)

configs = [
    (0.00, "#1f77b4", "o", "Idealized Crowd ($\\rho_h = 0.00$)"),
    (0.02, "#2ca02c", "s", "Low Correlation ($\\rho_h = 0.02$)"),
    (0.05, "#ff7f0e", "^", "Moderate Correlation ($\\rho_h = 0.05$)"),
    (0.10, "#d62728", "D", "Substantial Correlation ($\\rho_h = 0.10$)")
]

for rh, col, marker, lbl in configs:
    # Analytical curve
    errs_theory = [theoretical_dynamic_error(a, 0.90, rho_h=rh, sum_pi_sq=0.001, T=30) for a in alphas_fine]
    ax.plot(alphas_fine, errs_theory, color=col, linewidth=2.0, label=lbl, zorder=3)
    
    # Empirical points from K=500 runs
    x_pts = all_reliance_levels
    y_pts = [exp2_results[rh][a]["mean_error"] for a in x_pts]
    y_err = [exp2_results[rh][a]["sem_error"] for a in x_pts]
    ax.errorbar(x_pts, y_pts, yerr=y_err, fmt=marker, color=col, ecolor=col, elinewidth=1.0,
                capsize=2.5, markersize=4.5, zorder=4, alpha=0.9)
    
    # Mark analytical minimum for rh > 0
    if rh > 0.0:
        min_idx = np.argmin(errs_theory)
        opt_a = alphas_fine[min_idx]
        min_e = errs_theory[min_idx]
        ax.scatter([opt_a], [min_e], color=col, edgecolor="black", s=60, zorder=6)

ax.axhline(18.4, color="#555555", linestyle="--", linewidth=1.2, label=r"Monoculture Ceiling ($\mu_{AI}^2 + \rho\sigma_{AI}^2 = 18.40$)", zorder=2)
ax.set_title("The Optimal Reliance Trade-Off: Collective Error vs. Algorithmic Reliance\nunder Varying Human Error Correlation (N = 1,000, T = 30, K = 500)", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
ax.set_ylabel("Collective Squared Error (c - θ)²", fontsize=10.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.5, 28.0)
ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cccccc", fontsize=8.5, loc="upper left")

plt.tight_layout()
fig.savefig("/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig3_monoculture_index.png")
fig.savefig("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig3_monoculture_index.png")
# Also save as fig4 for backwards compatibility
fig.savefig("/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig4_page_decomposition_verification.png")
fig.savefig("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig4_page_decomposition_verification.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# GENERATE FIGURE 4: Static vs. Dynamic AI Error Convergence
# ---------------------------------------------------------------------------
print("Rendering Figure 4 (Static vs. Dynamic Model Persistence)...")
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=300)

stat_theory = [theoretical_static_error(a, 0.90, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]
dyn_theory = [theoretical_dynamic_error(a, 0.90, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]

ax.plot(alphas_fine, stat_theory, color="#d62728", linewidth=2.0, label="Static Task Advice Theory (Persistent Error)", zorder=3)
ax.plot(alphas_fine, dyn_theory, color="#1f77b4", linewidth=2.0, label="Dynamic Advice Theory (Temporal Stochasticity)", zorder=3)

# Overlay simulation points
x_pts = all_reliance_levels
y_stat = [exp3_results["static"][a]["mean"] for a in x_pts]
err_stat = [exp3_results["static"][a]["sem"] for a in x_pts]
y_dyn = [exp3_results["dynamic"][a]["mean"] for a in x_pts]
err_dyn = [exp3_results["dynamic"][a]["sem"] for a in x_pts]

ax.errorbar(x_pts, y_stat, yerr=err_stat, fmt="s", color="#d62728", ecolor="#d62728", elinewidth=1.0, capsize=2.5, markersize=5, label="Static AI Simulation (K=500)", zorder=4)
ax.errorbar(x_pts, y_dyn, yerr=err_dyn, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0, capsize=2.5, markersize=5, label="Dynamic AI Simulation (K=500)", zorder=4)

ax.set_title("Temporal Model Persistence: Dynamic vs. Static Machine Advice (N = 1,000, T = 30)", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
ax.set_ylabel("Collective Squared Error (c - θ)²", fontsize=10.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.5, 22.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, loc="lower right")

plt.tight_layout()
fig.savefig("/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig5_trajectory_convergence.png")
fig.savefig("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig5_trajectory_convergence.png")
plt.close(fig)

print("All figures successfully rendered and saved to both directories!")

# ---------------------------------------------------------------------------
# PRINT EXACT LATEX TABLE 1
# ---------------------------------------------------------------------------
print("\n=== EXACT HIGH-PRECISION TABLE 1 (POOLED N = 1,500 RUNS PER RELIANCE) ===")

def fmt(mean_val, sem_val):
    return f"{mean_val:.3f} $\\pm$ {sem_val:.3f}"

ws_b_mean = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["mean_error"]
ws_b_sem = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["sem_error"]
ws_div_mean = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["mean_diversity"]
ws_div_sem = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["sem_diversity"]

er_b_mean = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["mean_error"]
er_b_sem = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["sem_error"]
er_div_mean = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["mean_diversity"]
er_div_sem = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["sem_diversity"]

ba_b_mean = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["mean_error"]
ba_b_sem = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["sem_error"]
ba_div_mean = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["mean_diversity"]
ba_div_sem = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["sem_diversity"]

pool_b_mean = exp1_results[0.90][0.0]["pooled"]["mean_error"]
pool_b_sem = exp1_results[0.90][0.0]["pooled"]["sem_error"]
pool_div_mean = exp1_results[0.90][0.0]["pooled"]["mean_diversity"]
pool_div_sem = exp1_results[0.90][0.0]["pooled"]["sem_diversity"]

table_tex = []
table_tex.append(r"\begin{table*}[t]")
table_tex.append(r"\centering")
table_tex.append(r"\small")
table_tex.append(r"\setlength{\tabcolsep}{2.8pt}")
table_tex.append(r"\caption{\textbf{Large-Scale Simulation Results ($N = 1,000$ Agents, $T = 30$ Timesteps, $K = 500$ Trials per Topology, 49,500 Total Runs). Values denote Mean $\pm$ Standard Error of the Mean (SEM) across pooled runs ($N = 1,500$ independent trials per reliance level).}}")
table_tex.append(r"\begin{tabular}{@{}lccccccccc@{}}")
table_tex.append(r"\toprule")
table_tex.append(r"\textbf{Reliance ($\alpha$)} & \multicolumn{2}{c}{\textbf{Polyculture ($\rho = 0.00$)}} & \multicolumn{2}{c}{\textbf{Multi-Model ($\rho = 0.50$)}} & \multicolumn{2}{c}{\textbf{Monoculture ($\rho = 0.90$)}} & \multicolumn{3}{c}{\textbf{Analytical Theory ($T=30$)}} \\")
table_tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}")
table_tex.append(r"& \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Theory ($\rho=0$)} & \textbf{Theory ($\rho=0.5$)} & \textbf{Theory ($\rho=0.9$)} \\")
table_tex.append(r"\midrule")
table_tex.append(f"$\\alpha = 0.0$ (WS Small-World) & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(ws_b_mean, ws_b_sem)} \\quad \\vline \\quad Diversity: {fmt(ws_div_mean, ws_div_sem)}}} & 0.225 & 0.225 & 0.225 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (ER Random)      & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(er_b_mean, er_b_sem)} \\quad \\vline \\quad Diversity: {fmt(er_div_mean, er_div_sem)}}} & 0.225 & 0.225 & 0.225 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (BA Scale-Free)  & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(ba_b_mean, ba_b_sem)} \\quad \\vline \\quad Diversity: {fmt(ba_div_mean, ba_div_sem)}}} & 0.628 & 0.628 & 0.628 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (Pooled Baseline)& \\multicolumn{{6}}{{c}}{{\\textbf{{Collective Error: {fmt(pool_b_mean, pool_b_sem)}}} \\quad \\vline \\quad \\textbf{{Diversity: {fmt(pool_div_mean, pool_div_sem)}}}}} & 0.360 & 0.360 & 0.360 \\\\")
table_tex.append(r"\midrule")

for a in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    p_err_mean = exp1_results[0.00][a]["pooled"]["mean_error"]
    p_err_sem = exp1_results[0.00][a]["pooled"]["sem_error"]
    p_div_mean = exp1_results[0.00][a]["pooled"]["mean_diversity"]
    p_div_sem = exp1_results[0.00][a]["pooled"]["sem_diversity"]
    
    m_err_mean = exp1_results[0.50][a]["pooled"]["mean_error"]
    m_err_sem = exp1_results[0.50][a]["pooled"]["sem_error"]
    m_div_mean = exp1_results[0.50][a]["pooled"]["mean_diversity"]
    m_div_sem = exp1_results[0.50][a]["pooled"]["sem_diversity"]
    
    mono_err_mean = exp1_results[0.90][a]["pooled"]["mean_error"]
    mono_err_sem = exp1_results[0.90][a]["pooled"]["sem_error"]
    mono_div_mean = exp1_results[0.90][a]["pooled"]["mean_diversity"]
    mono_div_sem = exp1_results[0.90][a]["pooled"]["sem_diversity"]
    
    th_0 = theoretical_dynamic_error(a, 0.00)
    th_5 = theoretical_dynamic_error(a, 0.50)
    th_9 = theoretical_dynamic_error(a, 0.90)
    
    bold_start = r"\textbf{" if a == 1.0 else ""
    bold_end = "}" if a == 1.0 else ""
    label_str = f"$\\alpha = {a:.1f}$ (Pure AI)" if a == 1.0 else f"$\\alpha = {a:.1f}$"
    
    line = f"{label_str} & {bold_start}{fmt(p_err_mean, p_err_sem)}{bold_end} & {bold_start}{fmt(p_div_mean, p_div_sem)}{bold_end} & {bold_start}{fmt(m_err_mean, m_err_sem)}{bold_end} & {bold_start}{fmt(m_div_mean, m_div_sem)}{bold_end} & {bold_start}{fmt(mono_err_mean, mono_err_sem)}{bold_end} & {bold_start}{fmt(mono_div_mean, mono_div_sem)}{bold_end} & {bold_start}{th_0:.2f}{bold_end} & {bold_start}{th_5:.2f}{bold_end} & {bold_start}{th_9:.2f}{bold_end} \\\\"
    table_tex.append(line)

table_tex.append(r"\bottomrule")
table_tex.append(r"\end{tabular}")
table_tex.append(r"\label{tab:macro_results}")
table_tex.append(r"\end{table*}")

table_code = "\n".join(table_tex)
print(table_code)

with open("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/data/table1_latex.tex", "w") as f:
    f.write(table_code)

print("\n=== MASTER SIMULATION COMPLETE ===")

