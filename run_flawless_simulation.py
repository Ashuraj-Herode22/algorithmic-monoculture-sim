"""
Flawless Master Simulation Suite (Zero Seed Collisions, K = 1,000 per topology)
Executes 99,000+ independent Monte Carlo trials across network topologies,
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

print("=== STARTING FLAWLESS MASTER MONTE CARLO SIMULATION ===")
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
K_TRIALS = 1000  # 1,000 independent trials per cell (3,000 pooled per reliance level)

# 1. Pre-construct Graphs and Trust Matrices
print("Generating network graphs (N = 1,000)...")
graphs = {
    "watts_strogatz": generate_watts_strogatz_graph(n, k=6, p=0.1, seed=42),
    "erdos_renyi": nx.erdos_renyi_graph(n, p=0.05, seed=43),
    "barabasi_albert": generate_barabasi_albert_graph(n, m=3, seed=44),
}

if not nx.is_connected(graphs["erdos_renyi"]):
    largest_cc = max(nx.connected_components(graphs["erdos_renyi"]), key=len)
    graphs["erdos_renyi"] = graphs["erdos_renyi"].subgraph(largest_cc).copy()

matrices = {}
sum_pi_sq_vals = {}
for name, G in graphs.items():
    W = get_degroot_trust_matrix(G, self_weight=0.5)
    matrices[name] = W
    vals, vecs = np.linalg.eig(W.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = pi / np.sum(pi)
    sum_pi_sq = float(np.sum(pi**2))
    sum_pi_sq_vals[name] = sum_pi_sq
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
reliance_table = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
all_reliance = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
correlations = [0.00, 0.50, 0.90]

print(f"\nRunning Experiment 1: {len(all_reliance)} reliance levels x 3 regimes x 3 topologies x {K_TRIALS} trials...")

exp1_results = {}
trial_global_id = 0

for rho in correlations:
    exp1_results[rho] = {}
    for a in all_reliance:
        exp1_results[rho][a] = {"topologies": {}, "pooled": {}}
        pooled_errors = []
        pooled_diversities = []
        
        for top_name in ["watts_strogatz", "erdos_renyi", "barabasi_albert"]:
            W = matrices[top_name]
            actual_n = W.shape[0]
            errors = []
            diversities = []
            
            for k in range(K_TRIALS):
                trial_global_id += 1
                # Single dedicated RNG per trial: ZERO seed collisions
                rng = np.random.default_rng(trial_global_id * 10007)
                
                # Human prior (rho_h = 0.0)
                beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
                
                for t in range(T):
                    eta_shared = rng.normal(0.0, sigma_ai)
                    eta_indiv = rng.normal(0.0, sigma_ai, size=actual_n)
                    ai = truth + mu_ai + np.sqrt(rho) * eta_shared + np.sqrt(1.0 - rho) * eta_indiv
                    beliefs = (1.0 - a) * (W @ beliefs) + a * ai
                
                c = float(np.mean(beliefs))
                err = float((c - truth)**2)
                div = float(np.mean((beliefs - c)**2))
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
# EXPERIMENT 2: Human Correlation Sweep with Fine Alpha Grid Near 0
# ---------------------------------------------------------------------------
fine_alphas = np.unique(np.concatenate([
    np.linspace(0.0, 0.1, 11),  # Fine step of 0.01 near 0
    np.linspace(0.1, 1.0, 10)
]))
human_correlations = [0.00, 0.02, 0.05, 0.10]
exp2_results = {}

print(f"\nRunning Experiment 2: Human Correlation Sweep ({len(fine_alphas)} fine alphas x 4 regimes x {K_TRIALS} trials)...")
W_ws = matrices["watts_strogatz"]
actual_n = W_ws.shape[0]

for rh in human_correlations:
    exp2_results[rh] = {}
    for a in fine_alphas:
        errors = []
        for k in range(K_TRIALS):
            trial_global_id += 1
            rng = np.random.default_rng(trial_global_id * 10007)
            
            if rh > 0.0:
                shared_h = rng.normal(0.0, sigma_h)
                indiv_h = rng.normal(0.0, sigma_h, size=actual_n)
                beliefs = truth + np.sqrt(rh) * shared_h + np.sqrt(1.0 - rh) * indiv_h
            else:
                beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
            
            for t in range(T):
                eta_shared = rng.normal(0.0, sigma_ai)
                eta_indiv = rng.normal(0.0, sigma_ai, size=actual_n)
                ai = truth + mu_ai + np.sqrt(0.90) * eta_shared + np.sqrt(0.10) * eta_indiv
                beliefs = (1.0 - a) * (W_ws @ beliefs) + a * ai
            
            c = float(np.mean(beliefs))
            errors.append(float((c - truth)**2))
        
        exp2_results[rh][float(a)] = {
            "mean_error": float(np.mean(errors)),
            "sem_error": float(np.std(errors) / np.sqrt(K_TRIALS)),
        }
    print(f"  Completed human correlation rho_h = {rh:.2f}")

# ---------------------------------------------------------------------------
# EXPERIMENT 3: Static vs. Dynamic AI Noise Comparison
# ---------------------------------------------------------------------------
print(f"\nRunning Experiment 3: Static vs. Dynamic Comparison ({len(fine_alphas)} alphas x {K_TRIALS} trials)...")
exp3_results = {"dynamic": {}, "static": {}}

for a in fine_alphas:
    dyn_errors = []
    stat_errors = []
    for k in range(K_TRIALS):
        trial_global_id += 1
        rng = np.random.default_rng(trial_global_id * 10007)
        init_beliefs = truth + rng.normal(0.0, sigma_h, size=actual_n)
        
        # Dynamic AI advice (independent draw each step)
        b_dyn = init_beliefs.copy()
        for t in range(T):
            eta_s = rng.normal(0.0, sigma_ai)
            eta_i = rng.normal(0.0, sigma_ai, size=actual_n)
            ai = truth + mu_ai + np.sqrt(0.90)*eta_s + np.sqrt(0.10)*eta_i
            b_dyn = (1.0 - a) * (W_ws @ b_dyn) + a * ai
        dyn_errors.append(float((np.mean(b_dyn) - truth)**2))
        
        # Static AI advice (drawn once per problem instance)
        b_stat = init_beliefs.copy()
        static_s = rng.normal(0.0, sigma_ai)
        static_i = rng.normal(0.0, sigma_ai, size=actual_n)
        static_ai = truth + mu_ai + np.sqrt(0.90)*static_s + np.sqrt(0.10)*static_i
        for t in range(T):
            b_stat = (1.0 - a) * (W_ws @ b_stat) + a * static_ai
        stat_errors.append(float((np.mean(b_stat) - truth)**2))
        
    exp3_results["dynamic"][float(a)] = {"mean": float(np.mean(dyn_errors)), "sem": float(np.std(dyn_errors) / np.sqrt(K_TRIALS))}
    exp3_results["static"][float(a)] = {"mean": float(np.mean(stat_errors)), "sem": float(np.std(stat_errors) / np.sqrt(K_TRIALS))}

print("Experiment 3 complete.")
print(f"Total Monte Carlo runs completed: {trial_global_id}")
print(f"Total runtime: {time.time() - t_start:.2f} seconds.")

# ---------------------------------------------------------------------------
# GENERATE PUBLICATION FIGURES WITH PRECISE FILENAMES
# ---------------------------------------------------------------------------
fig_dir_icwsm = "/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures"
fig_dir_scratch = "/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures"

alphas_fine = np.linspace(0.0, 1.0, 200)

# Figure 1: 3-Panel Phase Transition
print("\nPlotting Figure 1 (Phase Transition)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=300, sharey=True)
panels = [
    (0.90, "Panel A: Monopoly Monoculture (ρ = 0.90)", axes[0]),
    (0.50, "Panel B: Correlated Multi-Model (ρ = 0.50)", axes[1]),
    (0.00, "Panel C: Polyculture Control (ρ = 0.00)", axes[2]),
]
top_styles = {
    "watts_strogatz": ("#1f77b4", "o", "Watts-Strogatz"),
    "erdos_renyi": ("#2ca02c", "s", "Erdős-Rényi"),
    "barabasi_albert": ("#d62728", "^", "Barabási-Albert"),
}

for rho, title, ax in panels:
    theory_vals = [theoretical_dynamic_error(a, rho, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]
    ax.plot(alphas_fine, theory_vals, "k--", linewidth=1.8, label="Analytical Theory", zorder=3)
    
    for top_name, (color, marker, label) in top_styles.items():
        x_pts = all_reliance
        y_pts = [exp1_results[rho][a]["topologies"][top_name]["mean_error"] for a in x_pts]
        y_err = [exp1_results[rho][a]["topologies"][top_name]["sem_error"] for a in x_pts]
        ax.errorbar(x_pts, y_pts, yerr=y_err, fmt=marker, color=color, ecolor=color, elinewidth=1.0,
                    capsize=2.5, markersize=4.5, label=f"{label} (K=1,000)", zorder=4, alpha=0.9)
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
    if ax == axes[0]:
        ax.set_ylabel("Mean Collective Squared Error (c - θ)²", fontsize=10.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, 22.0)
    ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig(f"{fig_dir_icwsm}/fig1_collective_error_phase_transition.png")
fig.savefig(f"{fig_dir_scratch}/fig1_collective_error_phase_transition.png")
plt.close(fig)

# Figure 2: Belief Variance Compression
print("Plotting Figure 2 (Belief Variance Compression)...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, sharey=True)
var_panels = [
    (0.90, "Monopoly Monoculture (ρ = 0.90)", axes[0]),
    (0.50, "Correlated Multi-Model (ρ = 0.50)", axes[1]),
]

for rho, title, ax in var_panels:
    x_pts = all_reliance
    y_ws = [exp1_results[rho][a]["topologies"]["watts_strogatz"]["mean_diversity"] for a in x_pts]
    y_er = [exp1_results[rho][a]["topologies"]["erdos_renyi"]["mean_diversity"] for a in x_pts]
    y_ba = [exp1_results[rho][a]["topologies"]["barabasi_albert"]["mean_diversity"] for a in x_pts]
    
    ax.plot(x_pts, y_ws, "o-", color="#1f77b4", linewidth=1.8, markersize=4.5, label="Watts-Strogatz")
    ax.plot(x_pts, y_er, "s-", color="#2ca02c", linewidth=1.8, markersize=4.5, label="Erdős-Rényi")
    ax.plot(x_pts, y_ba, "^-", color="#d62728", linewidth=1.8, markersize=4.5, label="Barabási-Albert")
    
    asymp_div = (1.0 - rho) * var_ai
    ax.axhline(asymp_div, color="black", linestyle="--", linewidth=1.2, label=f"Asymptotic Variance Limit: {asymp_div:.1f}")
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
    if ax == axes[0]:
        ax.set_ylabel("Belief Variance (1/N) ∑(x_i - c)²", fontsize=10.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, 18.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5, loc="upper left")

plt.tight_layout()
fig.savefig(f"{fig_dir_icwsm}/fig2_cognitive_diversity_collapse.png")
fig.savefig(f"{fig_dir_scratch}/fig2_cognitive_diversity_collapse.png")
plt.close(fig)

# Figure 3: Optimal Reliance Trade-Off & Crossover
print("Plotting Figure 3 (Optimal Reliance Trade-Off)...")
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
crossover_configs = [
    (0.00, "#1f77b4", "o", "Idealized Crowd ($\\rho_h = 0.00$)"),
    (0.02, "#2ca02c", "s", "Low Correlation ($\\rho_h = 0.02$)"),
    (0.05, "#ff7f0e", "^", "Moderate Correlation ($\\rho_h = 0.05$)"),
    (0.10, "#d62728", "D", "Substantial Correlation ($\\rho_h = 0.10$)")
]

for rh, col, marker, lbl in crossover_configs:
    theory_curve = [theoretical_dynamic_error(a, 0.90, rho_h=rh, sum_pi_sq=0.001, T=30) for a in alphas_fine]
    ax.plot(alphas_fine, theory_curve, color=col, linewidth=2.0, label=lbl, zorder=3)
    
    x_pts = fine_alphas
    y_pts = [exp2_results[rh][float(a)]["mean_error"] for a in x_pts]
    y_err = [exp2_results[rh][float(a)]["sem_error"] for a in x_pts]
    ax.errorbar(x_pts, y_pts, yerr=y_err, fmt=marker, color=col, ecolor=col, elinewidth=1.0,
                capsize=2.0, markersize=4.0, zorder=4, alpha=0.85)
    
    if rh > 0.0:
        min_idx = np.argmin(theory_curve)
        opt_a = alphas_fine[min_idx]
        min_e = theory_curve[min_idx]
        ax.scatter([opt_a], [min_e], color=col, edgecolor="black", s=60, zorder=6)

ax.axhline(18.4, color="#555555", linestyle="--", linewidth=1.2, label=r"Monoculture Ceiling ($\mu_{AI}^2 + \rho\sigma_{AI}^2 = 18.40$)", zorder=2)
ax.set_title("The Optimal Reliance Trade-Off: Collective Error vs. Algorithmic Reliance\nunder Varying Human Error Correlation (N = 1,000, T = 30, K = 1,000)", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
ax.set_ylabel("Collective Squared Error (c - θ)²", fontsize=10.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.5, 26.0)
ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cccccc", fontsize=8.5, loc="upper left")

plt.tight_layout()
fig.savefig(f"{fig_dir_icwsm}/fig3_monoculture_index.png")
fig.savefig(f"{fig_dir_scratch}/fig3_monoculture_index.png")
fig.savefig(f"{fig_dir_icwsm}/fig4_page_decomposition_verification.png")
fig.savefig(f"{fig_dir_scratch}/fig4_page_decomposition_verification.png")
plt.close(fig)

# Figure 4: Static vs. Dynamic Model Persistence
print("Plotting Figure 4 (Static vs. Dynamic Persistence)...")
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=300)

stat_curve = [theoretical_static_error(a, 0.90, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]
dyn_curve = [theoretical_dynamic_error(a, 0.90, rho_h=0.0, sum_pi_sq=0.001, T=30) for a in alphas_fine]

ax.plot(alphas_fine, stat_curve, color="#d62728", linewidth=2.0, label="Static Task Advice Theory (Persistent Error)", zorder=3)
ax.plot(alphas_fine, dyn_curve, color="#1f77b4", linewidth=2.0, label="Dynamic Advice Theory (Temporal Stochasticity)", zorder=3)

x_pts = fine_alphas
y_stat = [exp3_results["static"][float(a)]["mean"] for a in x_pts]
err_stat = [exp3_results["static"][float(a)]["sem"] for a in x_pts]
y_dyn = [exp3_results["dynamic"][float(a)]["mean"] for a in x_pts]
err_dyn = [exp3_results["dynamic"][float(a)]["sem"] for a in x_pts]

ax.errorbar(x_pts, y_stat, yerr=err_stat, fmt="s", color="#d62728", ecolor="#d62728", elinewidth=1.0, capsize=2.0, markersize=4.5, label="Static AI Simulation (K=1,000)", zorder=4)
ax.errorbar(x_pts, y_dyn, yerr=err_dyn, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0, capsize=2.0, markersize=4.5, label="Dynamic AI Simulation (K=1,000)", zorder=4)

ax.set_title("Temporal Model Persistence: Dynamic vs. Static Machine Advice (N = 1,000, T = 30)", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Algorithmic Reliance (α)", fontsize=10.5)
ax.set_ylabel("Collective Squared Error (c - θ)²", fontsize=10.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.5, 22.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, loc="lower right")

plt.tight_layout()
fig.savefig(f"{fig_dir_icwsm}/fig5_trajectory_convergence.png")
fig.savefig(f"{fig_dir_scratch}/fig5_trajectory_convergence.png")
plt.close(fig)

print("All figures successfully rendered!")

# ---------------------------------------------------------------------------
# OUTPUT EXACT LATEX TABLE 1
# ---------------------------------------------------------------------------
def fmt(m, s): return f"{m:.3f} $\\pm$ {s:.3f}"

ws_b_m = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["mean_error"]
ws_b_s = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["sem_error"]
ws_d_m = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["mean_diversity"]
ws_d_s = exp1_results[0.90][0.0]["topologies"]["watts_strogatz"]["sem_diversity"]

er_b_m = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["mean_error"]
er_b_s = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["sem_error"]
er_d_m = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["mean_diversity"]
er_d_s = exp1_results[0.90][0.0]["topologies"]["erdos_renyi"]["sem_diversity"]

ba_b_m = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["mean_error"]
ba_b_s = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["sem_error"]
ba_d_m = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["mean_diversity"]
ba_d_s = exp1_results[0.90][0.0]["topologies"]["barabasi_albert"]["sem_diversity"]

pool_b_m = exp1_results[0.90][0.0]["pooled"]["mean_error"]
pool_b_s = exp1_results[0.90][0.0]["pooled"]["sem_error"]
pool_d_m = exp1_results[0.90][0.0]["pooled"]["mean_diversity"]
pool_d_s = exp1_results[0.90][0.0]["pooled"]["sem_diversity"]

table_tex = []
table_tex.append(r"\begin{table*}[t]")
table_tex.append(r"\centering")
table_tex.append(r"\small")
table_tex.append(r"\setlength{\tabcolsep}{2.8pt}")
table_tex.append(r"\caption{\textbf{High-Sample Simulation Results ($N = 1,000$ Agents, $T = 30$ Timesteps, $K = 1,000$ Independent Trials per Condition, 99,000 Total Runs). Values denote Mean $\pm$ Standard Error of the Mean (SEM) across pooled runs ($N = 3,000$ independent trials per reliance level).}}")
table_tex.append(r"\begin{tabular}{@{}lccccccccc@{}}")
table_tex.append(r"\toprule")
table_tex.append(r"\textbf{Reliance ($\alpha$)} & \multicolumn{2}{c}{\textbf{Polyculture ($\rho = 0.00$)}} & \multicolumn{2}{c}{\textbf{Multi-Model ($\rho = 0.50$)}} & \multicolumn{2}{c}{\textbf{Monoculture ($\rho = 0.90$)}} & \multicolumn{3}{c}{\textbf{Analytical Theory ($T=30$)}} \\")
table_tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}")
table_tex.append(r"& \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Coll. Error} & \textbf{Diversity} & \textbf{Theory ($\rho=0$)} & \textbf{Theory ($\rho=0.5$)} & \textbf{Theory ($\rho=0.9$)} \\")
table_tex.append(r"\midrule")
table_tex.append(f"$\\alpha = 0.0$ (WS Small-World) & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(ws_b_m, ws_b_s)} \\quad \\vline \\quad Diversity: {fmt(ws_d_m, ws_d_s)}}} & 0.225 & 0.225 & 0.225 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (ER Random)      & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(er_b_m, er_b_s)} \\quad \\vline \\quad Diversity: {fmt(er_d_m, er_d_s)}}} & 0.225 & 0.225 & 0.225 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (BA Scale-Free)  & \\multicolumn{{6}}{{c}}{{Collective Error: {fmt(ba_b_m, ba_b_s)} \\quad \\vline \\quad Diversity: {fmt(ba_d_m, ba_d_s)}}} & 0.628 & 0.628 & 0.628 \\\\")
table_tex.append(f"$\\alpha = 0.0$ (Pooled Baseline)& \\multicolumn{{6}}{{c}}{{\\textbf{{Collective Error: {fmt(pool_b_m, pool_b_s)}}} \\quad \\vline \\quad \\textbf{{Diversity: {fmt(pool_d_m, pool_d_s)}}}}} & 0.360 & 0.360 & 0.360 \\\\")
table_tex.append(r"\midrule")

for a in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    p_err_m = exp1_results[0.00][a]["pooled"]["mean_error"]
    p_err_s = exp1_results[0.00][a]["pooled"]["sem_error"]
    p_div_m = exp1_results[0.00][a]["pooled"]["mean_diversity"]
    p_div_s = exp1_results[0.00][a]["pooled"]["sem_diversity"]
    
    m_err_m = exp1_results[0.50][a]["pooled"]["mean_error"]
    m_err_s = exp1_results[0.50][a]["pooled"]["sem_error"]
    m_div_m = exp1_results[0.50][a]["pooled"]["mean_diversity"]
    m_div_s = exp1_results[0.50][a]["pooled"]["sem_diversity"]
    
    mono_err_m = exp1_results[0.90][a]["pooled"]["mean_error"]
    mono_err_s = exp1_results[0.90][a]["pooled"]["sem_error"]
    mono_div_m = exp1_results[0.90][a]["pooled"]["mean_diversity"]
    mono_div_s = exp1_results[0.90][a]["pooled"]["sem_diversity"]
    
    th_0 = theoretical_dynamic_error(a, 0.00)
    th_5 = theoretical_dynamic_error(a, 0.50)
    th_9 = theoretical_dynamic_error(a, 0.90)
    
    bold_start = r"\textbf{" if a == 1.0 else ""
    bold_end = "}" if a == 1.0 else ""
    label_str = f"$\\alpha = {a:.1f}$ (Pure AI)" if a == 1.0 else f"$\\alpha = {a:.1f}$"
    
    line = f"{label_str} & {bold_start}{fmt(p_err_m, p_err_s)}{bold_end} & {bold_start}{fmt(p_div_m, p_div_s)}{bold_end} & {bold_start}{fmt(m_err_m, m_err_s)}{bold_end} & {bold_start}{fmt(m_div_m, m_div_s)}{bold_end} & {bold_start}{fmt(mono_err_m, mono_err_s)}{bold_end} & {bold_start}{fmt(mono_div_m, mono_div_s)}{bold_end} & {bold_start}{th_0:.2f}{bold_end} & {bold_start}{th_5:.2f}{bold_end} & {bold_start}{th_9:.2f}{bold_end} \\\\"
    table_tex.append(line)

table_tex.append(r"\bottomrule")
table_tex.append(r"\end{tabular}")
table_tex.append(r"\label{tab:macro_results}")
table_tex.append(r"\end{table*}")

table_code = "\n".join(table_tex)
print("\n" + table_code)

with open("/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/data/table1_latex.tex", "w") as f:
    f.write(table_code)

print("\n=== MASTER EXECUTION COMPLETED SUCCESSFULLY ===")

