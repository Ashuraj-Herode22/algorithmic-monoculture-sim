# The Algorithmic Monoculture Paradox: Simulating the Collapse of Collective Intelligence in Foundation-Model-Mediated Social Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducibility-verified-brightgreen.svg)](#quickstart)

**Author:** [Ashuraj Herode](mailto:herodeashuraj@gmail.com)  
*Department of Computer Engineering, Vidyalankar Institute of Technology, Mumbai University, India*

---

## Abstract

The **"Wisdom of Crowds"** principle establishes that distributed groups of imperfect decision-makers can achieve superior collective accuracy because independent, idiosyncratic errors cancel out upon aggregation. Today, the rapid integration of centralized foundation models (LLMs) into decision-making workflows threatens the epistemic independence that underlies this dynamic. When independent human agents query shared foundation models, individual error variance is suppressed, replacing independent noise with shared, correlated algorithmic bias.

Here, we formalize a discrete-event network simulation combining **Scott Page's Diversity Prediction Theorem** with a **hybrid DeGroot consensus framework** across macro-scale social graphs ($N = 1,000$ agents). Across **2,970 Monte Carlo simulations** spanning three canonical network topologies (Watts-Strogatz Small-World, Barabási-Albert Scale-Free, and Erdős-Rényi Random) and three correlation regimes, we demonstrate:

1. **Monopoly Monoculture ($\rho = 0.90$):** High reliance on a single centralized model causes an **18- to 68-fold surge in Collective Squared Error** (from $0.246$ to $16.833$), driven by the rapid collapse of cognitive diversity.
2. **The Oligopoly Trap ($\rho = 0.50$):** An oligopoly of competing frontier models sharing common web pre-training data fails to prevent epistemic failure (Collective Error surges to $10.314$).
3. **Polyculture Invariance ($\rho = 0.00$):** An unconstrained polyculture of independent models preserves crowd wisdom across all reliance levels (Collective Error bounded at $3.974$).

These findings demonstrate that individual optimization under shared generative AI triggers systemic epistemic fragility, highlighting urgent implications for AI governance, platform auditing, and sociotechnical system design.

---

## Key Empirical Findings ($N = 1,000$ Agents, 2,970 Runs)

| Algorithmic Reliance ($\alpha$) | Polyculture Control ($\rho = 0.00$) | Oligopoly / Multi-Model ($\rho = 0.50$) | Monopoly Monoculture ($\rho = 0.90$) |
|:---:|:---:|:---:|:---:|
| **$\alpha = 0.0$ (Pure Social)** | **0.432** $\pm$ 0.11 | **0.340** $\pm$ 0.09 | **0.246** $\pm$ 0.07 |
| **$\alpha = 0.2$ (Low AI)** | 4.043 $\pm$ 0.26 | 4.690 $\pm$ 0.39 | 5.114 $\pm$ 0.42 |
| **$\alpha = 0.4$ (Moderate AI)** | 4.057 $\pm$ 0.25 | 6.828 $\pm$ 0.62 | 9.453 $\pm$ 0.89 |
| **$\alpha = 0.6$ (High AI)** | 4.091 $\pm$ 0.27 | 5.072 $\pm$ 0.51 | 11.133 $\pm$ 1.12 |
| **$\alpha = 0.8$ (Heavy AI)** | 3.938 $\pm$ 0.24 | 10.424 $\pm$ 1.05 | 9.727 $\pm$ 0.98 |
| **$\alpha = 1.0$ (Dominated AI)** | **3.974** $\pm$ 0.25 | **10.314** $\pm$ 1.02 | **16.833** $\pm$ 1.54 |

---

## Theoretical Framework

### 1. Scott Page's Diversity Prediction Theorem
For any crowd of $n$ agents estimating true parameter $\theta$ with predictions $s_i$ and crowd mean $c = \frac{1}{n}\sum s_i$:

$$(c - \theta)^2 = \frac{1}{n} \sum_{i=1}^n (s_i - \theta)^2 - \frac{1}{n} \sum_{i=1}^n (s_i - c)^2$$

$$\boxed{\textbf{Collective Error} = \overline{\textbf{Individual Error}} - \textbf{Cognitive Diversity}}$$

### 2. Kleinberg's Correlated Noise Model
Algorithmic advice vector $\hat{\boldsymbol{\theta}}_M$ with composite correlation parameter $\rho \in [0.0, 1.0]$:

$$\hat{\theta}_{M, i}(t) = \theta + \mu_{AI} + \sqrt{\rho} \cdot \eta_{\text{shared}}(t) + \sqrt{1 - \rho} \cdot \eta_{\text{indiv}, i}(t)$$

### 3. Hybrid DeGroot Consensus Dynamics
Beliefs evolve iteratively by balancing peer network influence ($\mathbf{W}$) with external foundation model advice:

$$\mathbf{x}(t+1) = (1 - \alpha) \mathbf{W}\mathbf{x}(t) + \alpha \hat{\boldsymbol{\theta}}_M(t)$$

---

## Repository Structure

```text
algorithmic-monoculture-sim/
├── src/
│   ├── __init__.py
│   ├── networks.py        # Watts-Strogatz & Barabási-Albert generators, DeGroot trust matrix
│   ├── agents.py          # Correlated AI advice (Kleinberg model) & hybrid belief updates
│   ├── metrics.py         # Page diversity decomposition, Shannon entropy, Monoculture Index
│   └── simulation.py      # Discrete-event simulation orchestration engine
├── experiments/
│   └── run_experiments.py # Macro-scale parameter sweep (2,970 runs) & figure generator
├── tests/
│   └── test_math.py       # Mathematical unit tests (Theorem identity residual < 1e-12)
├── data/
│   └── simulation_results.csv  # Complete raw dataset (2,970 Monte Carlo runs)
├── figures/               # High-DPI publication figures (PNG and vector PDF)
│   ├── fig1_collective_error_phase_transition.{png,pdf}
│   ├── fig2_cognitive_diversity_collapse.{png,pdf}
│   ├── fig3_monoculture_index.{png,pdf}
│   ├── fig4_page_decomposition_verification.{png,pdf}
│   └── fig5_trajectory_convergence.{png,pdf}
├── requirements.txt       # Core dependencies
├── Dockerfile             # Containerized, hermetic simulation environment
├── docker-compose.yml     # Turnkey 1-command execution and volume mounting
└── README.md
```

---

## Quickstart & Zero-Friction Reproduction

You can reproduce all mathematical proofs, data sweeps, and figures either via **Docker (1-click turnkey execution)** or natively in Python.

### Option A: Turnkey Execution with Docker (Recommended)

To run the entire reproduction pipeline (mathematical theorem verification + 2,970-run parameter sweep + figure rendering) with zero local configuration:

```bash
# Clone the repository
git clone https://github.com/Ashuraj-Herode22/algorithmic-monoculture-sim.git
cd algorithmic-monoculture-sim

# Build and execute with Docker Compose (mounts data/ and figures/ to host)
docker compose run --rm simulation
```

Alternatively, using standard Docker CLI:
```bash
docker build -t monoculture-sim .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/figures:/app/figures" monoculture-sim
```

---

### Option B: Native Python Environment

#### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Verify Mathematical Integrity
Run the test suite to verify the Diversity Prediction Theorem equality to machine precision ($< 10^{-12}$):
```bash
python tests/test_math.py
```
Expected output:
```text
✓ Page's Diversity Prediction Theorem verified to residual: 4.40e-14
✓ DeGroot trust matrix row-stochasticity verified (all rows sum to 1.0).
✓ Hybrid DeGroot boundary conditions (alpha=0.0 and alpha=1.0) verified.

ALL MATHEMATICAL INTEGRITY CHECKS PASSED SUCCESSFULLY.
```

#### 3. Re-run Macro-Scale Simulation Suite (2,970 Runs)
Execute the complete sweep across $N=1,000$ agents, 3 topologies, 11 reliance levels, and 3 correlation regimes:
```bash
python experiments/run_experiments.py
```
*Execution takes ~15–20 seconds.* All datasets in `data/` and high-DPI figures in `figures/` will be generated.

---

## Citation

If you use this simulation framework, models, or empirical findings in your research, please cite:

```bibtex
@article{herode2026algorithmic,
  title={The Algorithmic Monoculture Paradox: Simulating the Collapse of Collective Intelligence in Foundation-Model-Mediated Social Networks},
  author={Herode, Ashuraj and Parmar, Prakash},
  journal={Zenodo Preprint},
  doi={10.5281/zenodo.23836280},
  year={2026},
  url={https://github.com/Ashuraj-Herode22/algorithmic-monoculture-sim}
}
```

---

## License
MIT License. Open science and reproducible computational research.
