# The Algorithmic Monoculture Paradox: Simulating Epistemic Degradation in Foundation-Model-Mediated Social Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Author:** [Ashuraj Herode](mailto:herodeashuraj@gmail.com)  
*Department of Computer Engineering, Vidyalankar Institute of Technology, University of Mumbai, India*  
**Co-Author / Advisor:** Dr. Prakash Parmar (*Vidyalankar Institute of Technology, University of Mumbai*)  
**Zenodo DOI:** [10.5281/zenodo.23836280](https://doi.org/10.5281/zenodo.23836280)

---

## Abstract

The **"Wisdom of Crowds"** principle establishes that distributed groups of imperfect decision-makers can achieve superior collective accuracy because independent, idiosyncratic errors cancel out upon aggregation. Today, the rapid integration of centralized foundation models (LLMs) into decision-making workflows threatens the epistemic independence that underlies this dynamic. When independent human agents query shared foundation models, individual error variance is suppressed, replacing independent noise with shared, correlated algorithmic bias.

Here, we formalize a discrete-event network simulation combining **Scott Page's Diversity Prediction Theorem** (Krogh-Vedelsby ambiguity decomposition) with a **hybrid DeGroot consensus framework** across macro-scale social graphs ($N = 1,000$ agents). Across **2,970 Monte Carlo simulations** spanning three canonical network topologies (Watts-Strogatz Small-World, Barabási-Albert Scale-Free, and Erdős-Rényi Random) and three correlation regimes, we demonstrate:

1. **Monopoly Monoculture ($\rho = 0.90$):** High reliance on a single centralized model causes an **average 50-fold surge in Collective Squared Error** (escalating by over $+16$ squared error units from a pooled baseline of $0.339 \pm 0.07$ to $16.833 \pm 2.05$).
2. **The Oligopoly Trap ($\rho = 0.50$):** An oligopoly of competing frontier models sharing common web pre-training data fails to prevent epistemic degradation (Collective Error surges to $10.314 \pm 1.59$).
3. **Polyculture Bias Floor ($\rho = 0.00$):** While an unconstrained polyculture of independent models eliminates correlated variance ($\rho\sigma_{AI}^2$), collective error remains bounded near the directional bias floor ($3.974 \pm 0.05$)—approximately 12-fold higher than the unassisted crowd.
4. **The Human Correlation Crossover ($\rho_h^*$):** Against an idealized unbiased crowd ($\rho_h = 0$), monoculture degrades collective judgment. However, if human decision-makers share merely **8.1%** correlated error through common cognitive biases ($\rho_h \ge 0.081$), AI monoculture yields superior collective accuracy over the human crowd. For polyculture, the crossover occurs at just **1.7%** ($\rho_h^* \ge 0.017$).
5. **Topology Invariance:** Because uniform external advice passes directly through any row-stochastic network matrix ($\mathbf{W}\mathbf{1} = \mathbf{1}$), the AI-driven error component is strictly topology-invariant.

---

## Key Empirical Findings ($N = 1,000$ Agents, $T = 30$ Timesteps, 2,970 Total Runs)

Values denote **Mean $\pm$ Standard Error of the Mean (SEM)** across independent trials ($N = 90$ runs per reliance level).

| Algorithmic Reliance ($\alpha$) | Polyculture Control ($\rho = 0.00$) | Oligopoly ($\rho = 0.50$) | Monopoly Monoculture ($\rho = 0.90$) | Theory ($\rho=0.0$) | Theory ($\rho=0.5$) | Theory ($\rho=0.9$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\alpha = 0.0$ (WS Small-World)** | \multicolumn{3}{c|}{Coll. Error: $0.224 \pm 0.069$ \| Diversity: $1.444 \pm 0.004$} | 0.225 | 0.225 | 0.225 |
| **$\alpha = 0.0$ (ER Random)** | \multicolumn{3}{c|}{Coll. Error: $0.216 \pm 0.062$ \| Diversity: $0.000 \pm 0.000$} | 0.225 | 0.225 | 0.225 |
| **$\alpha = 0.0$ (BA Scale-Free)** | \multicolumn{3}{c|}{Coll. Error: $0.578 \pm 0.164$ \| Diversity: $0.000 \pm 0.000$} | 0.628 | 0.628 | 0.628 |
| **$\alpha = 0.0$ (Pooled Baseline)** | \multicolumn{3}{c|}{\textbf{Coll. Error: $0.339 \pm 0.072$ \| Diversity: $0.481 \pm 0.072$}} | 0.360 | 0.360 | 0.360 |
| **$\alpha = 0.1$** | 3.673 $\pm$ 0.015 | 4.358 $\pm$ 0.281 | 4.165 $\pm$ 0.373 | 3.67 | 4.09 | 4.42 |
| **$\alpha = 0.2$** | 4.043 $\pm$ 0.019 | 4.690 $\pm$ 0.464 | 5.114 $\pm$ 0.593 | 3.99 | 4.88 | 5.59 |
| **$\alpha = 0.4$** | 4.057 $\pm$ 0.030 | 6.828 $\pm$ 0.660 | 9.453 $\pm$ 0.969 | 4.00 | 6.00 | 7.60 |
| **$\alpha = 0.6$** | 4.091 $\pm$ 0.029 | 5.072 $\pm$ 0.527 | 11.133 $\pm$ 1.380 | 4.00 | 7.43 | 10.17 |
| **$\alpha = 0.8$** | 3.938 $\pm$ 0.041 | 10.424 $\pm$ 1.791 | 9.727 $\pm$ 1.342 | 4.00 | 9.33 | 13.60 |
| **$\alpha = 1.0$ (Pure AI)** | **3.974 $\pm$ 0.052** | **10.314 $\pm$ 1.586** | **16.833 $\pm$ 2.046** | **4.00** | **12.00** | **18.40** |

---

## Theoretical Framework

### 1. Scott Page's Diversity Prediction Theorem
For any crowd of $N$ agents estimating true parameter $\theta$ with individual predictions $s_i$ and crowd mean $c = \frac{1}{N}\sum s_i$, collective squared error decomposes exactly (Krogh & Vedelsby 1995; Page 2007):

$$(c - \theta)^2 = \frac{1}{N} \sum_{i=1}^N (s_i - \theta)^2 - \frac{1}{N} \sum_{i=1}^N (s_i - c)^2$$

$$\boxed{\textbf{Collective Error} = \overline{\textbf{Individual Error}} - \textbf{Cognitive Diversity}}$$

### 2. Iterated Hybrid DeGroot Consensus Dynamics
At each timestep $t$, the belief vector $\mathbf{x}(t) \in \mathbb{R}^N$ updates via:

$$\mathbf{x}(t+1) = (1 - \alpha) \mathbf{W}\mathbf{x}(t) + \alpha \hat{\boldsymbol{\theta}}_M(t)$$

where $\mathbf{W}$ is any row-stochastic trust matrix ($\mathbf{W}\mathbf{1} = \mathbf{1}$) and external AI advice is parameterized by:

$$\hat{\theta}_{M, i}(t) = \theta + \mu_{AI} + \sqrt{\rho}\eta_{\text{shared}}(t) + \sqrt{1 - \rho}\eta_{\text{indiv}, i}(t)$$

### 3. Mean-Field Decomposition & Topology Invariance Proof
Decomposing the state vector as $\mathbf{x}(t) = m(t)\mathbf{1} + \mathbf{d}(t)$, where $m(t) \equiv \boldsymbol{\pi}^T \mathbf{x}(t)$ is weighted by the stationary vector $\boldsymbol{\pi}$ of $\mathbf{W}$ ($\boldsymbol{\pi}^T \mathbf{W} = \boldsymbol{\pi}^T, \boldsymbol{\pi}^T \mathbf{1} = 1$). 

Because uniform advice passes directly through any row-stochastic matrix ($\mathbf{W}\mathbf{1} = \mathbf{1}$), unrolling the recurrence yields the **finite-time expected collective error**:

$$E_{\text{theory}}(\alpha, \rho, T) = \left[\mu_{AI} (1 - (1 - \alpha)^T)\right]^2 + \frac{\alpha(1 - (1 - \alpha)^{2T})}{2 - \alpha} \rho \sigma_{AI}^2$$

As $T \to \infty$, this converges to the steady-state formula:

$$\mathbb{E}[(c^* - \theta)^2] = \mu_{AI}^2 + \frac{\alpha}{2 - \alpha}\rho\sigma_{AI}^2$$

### 4. Baseline Human Variance by Network Topology
For unassisted human deliberation ($\alpha = 0$), the expected error depends directly on topology through the stationary distribution weights:

$$\text{Var}(c_{\text{human}}) = \sigma_{\text{human}}^2 \sum_{i=1}^N \pi_i^2$$

- **Watts-Strogatz & Erdős-Rényi:** $\pi_i \approx 1/N$, predicting $\text{Var} = 225 \times 0.001 = 0.225$ (empirical: $0.224, 0.216$).
- **Barabási-Albert Scale-Free:** Hubs concentrate weight ($\sum \pi_i^2 \approx 0.0028$), predicting $\text{Var} = 225 \times 0.0028 = 0.628$ (empirical: $0.578$, formalizing Golub & Jackson 2010).

### 5. The Human Correlation Crossover ($\rho_h^*$)
If human decision-makers share pairwise correlated error $\rho_h \in [0, 1]$ (e.g., common media framing, shared heuristics):

$$\text{MSE}_{\text{human}}(\rho_h) = \rho_h \sigma_{\text{human}}^2 + (1 - \rho_h)\sigma_{\text{human}}^2 \sum_{i=1}^N \pi_i^2 \approx 0.225 + 224.775\,\rho_h$$

Equating this to the steady-state error of algorithmic advice reveals the crossover threshold $\rho_h^*$:
- **Monopoly Monoculture ($\text{MSE} = 18.40$):** $\rho_h^* = \frac{18.40 - 0.225}{224.775} \approx 8.1\%$.
- **Polyculture Control ($\text{MSE} = 4.00$):** $\rho_h^* = \frac{4.00 - 0.225}{224.775} \approx 1.7\%$.

When human crowd error correlation exceeds $8.1\%$, foundation model monoculture actually outperforms the human crowd.

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

## Quickstart & Reproduction

You can reproduce all mathematical proofs, data sweeps, and figures either via **Docker** or natively in Python.

### Option A: Turnkey Execution with Docker (Recommended)

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
  title={The Algorithmic Monoculture Paradox: Simulating Epistemic Degradation in Foundation-Model-Mediated Social Networks},
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
