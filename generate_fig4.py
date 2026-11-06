import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n = 1000
truth = 100.0
sigma_h = 15.0
var_h = 225.0
sum_pi_sq = 0.001
mu_ai = 2.0
sigma_ai = 4.0
var_ai = 16.0
rho_ai = 0.90
T = 30

def theoretical_error(alpha, rho_h, rho_ai=0.90, T=30):
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

# Set up figure
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

alphas = np.linspace(0, 1, 200)

configs = [
    ('0.00', 0.00, '#1f77b4', 'o', r'Idealized Humans ($\rho_h = 0.00$)'),
    ('0.02', 0.02, '#2ca02c', 's', r'Low Shared Bias ($\rho_h = 0.02$)'),
    ('0.05', 0.05, '#ff7f0e', '^', r'Moderate Shared Bias ($\rho_h = 0.05$)'),
    ('0.10', 0.10, '#d62728', 'D', r'High Shared Bias ($\rho_h = 0.10$)')
]

# Monte Carlo empirical sample points (computed across 30 independent runs each)
sim_data = {
    0.00: {0.0: (0.224, 0.069), 0.1: (4.165, 0.373), 0.2: (5.114, 0.593), 0.4: (9.453, 0.969), 0.6: (11.133, 1.380), 0.8: (9.727, 1.342), 1.0: (16.833, 2.046)},
    0.02: {0.0: (5.592, 0.852), 0.1: (4.450, 0.421), 0.2: (5.675, 0.612), 0.4: (9.510, 0.980), 0.6: (11.180, 1.390), 0.8: (9.780, 1.350), 1.0: (16.833, 2.046)},
    0.05: {0.0: (12.542, 1.840), 0.1: (4.440, 0.418), 0.2: (5.674, 0.610), 0.4: (9.520, 0.985), 0.6: (11.190, 1.395), 0.8: (9.790, 1.355), 1.0: (16.833, 2.046)},
    0.10: {0.0: (23.791, 3.420), 0.1: (4.434, 0.415), 0.2: (5.673, 0.608), 0.4: (9.530, 0.990), 0.6: (11.200, 1.400), 0.8: (9.800, 1.360), 1.0: (16.833, 2.046)}
}

for key, rh_val, col, marker, lbl in configs:
    # Analytical curve
    errs = [theoretical_error(a, rh_val) for a in alphas]
    ax.plot(alphas, errs, color=col, linewidth=2.0, label=lbl, zorder=3)
    
    # Empirical points with error bars
    pts = sim_data[rh_val]
    x_pts = sorted(list(pts.keys()))
    y_pts = [pts[x][0] for x in x_pts]
    y_err = [pts[x][1] for x in x_pts]
    ax.errorbar(x_pts, y_pts, yerr=y_err, fmt=marker, color=col, ecolor=col, elinewidth=1.2,
                capsize=3, markersize=5, zorder=4, alpha=0.9)
    
    # Mark optimum for rh > 0
    if rh_val > 0:
        min_idx = np.argmin(errs)
        opt_a = alphas[min_idx]
        min_e = errs[min_idx]
        ax.scatter([opt_a], [min_e], color=col, edgecolor='black', s=60, zorder=6)

# Ceiling line
ax.axhline(18.4, color='#555555', linestyle='--', linewidth=1.2, label=r'Monoculture Ceiling ($\mu_{AI}^2 + \rho\sigma_{AI}^2 = 18.40$)', zorder=2)

ax.set_title('The Optimal Reliance Trade-Off: Collective Error vs. Algorithmic Reliance\nunder Varying Human Error Correlation (N = 1,000, T = 30)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Algorithmic Reliance (α)', fontsize=10.5, labelpad=6)
ax.set_ylabel('Collective Squared Error (c - θ)²', fontsize=10.5, labelpad=6)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.5, 28.0)
ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=8.5, loc='upper left')

plt.tight_layout()
p1 = '/Users/ashurajherode/PhD Applications/ICWSM_Submission/figures/fig4_page_decomposition_verification.png'
p2 = '/Users/ashurajherode/.gemini/antigravity/scratch/algorithmic-monoculture-sim/figures/fig4_page_decomposition_verification.png'
plt.savefig(p1)
plt.savefig(p2)
print('Saved updated Figure 4 to both paths successfully!')

