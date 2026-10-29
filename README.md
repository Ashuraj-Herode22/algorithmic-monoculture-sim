# The Algorithmic Monoculture Paradox: Simulating the Collapse of Collective Intelligence in Foundation-Model-Mediated Social Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Computational framework and simulation engine investigating how increasing reliance on centralized foundation models impacts the "Wisdom of Crowds" across complex network topologies (Watts-Strogatz Small-World vs. Barabási-Albert Scale-Free).

---

## Overview

The **Diversity Prediction Theorem** (Page, 2007) establishes that crowd wisdom fundamentally depends on individual error variance:

$$\text{Collective Squared Error} = \overline{\text{Individual Squared Error}} - \text{Cognitive Diversity}$$

When independent agents query shared foundation models (Kleinberg & Raghavan, 2021; Bommasani et al., 2024), error independence breaks down. This repository provides a vectorized discrete-event network simulation modeling belief evolution under hybrid peer-algorithmic consensus dynamics:

$$\mathbf{x}(t+1) = (1 - \alpha) \mathbf{W}\mathbf{x}(t) + \alpha \hat{\boldsymbol{\theta}}_M(t)$$

where:
* $\mathbf{W}$ is a row-stochastic social influence matrix.
* $\hat{\boldsymbol{\theta}}_M$ is the correlated algorithmic advice vector.
* $\alpha \in [0.0, 1.0]$ represents the degree of algorithmic reliance.

---

## Repository Structure

```text
algorithmic-monoculture-sim/
├── src/
│   ├── __init__.py
│   ├── networks.py        # Small-World & Scale-Free graph generators; DeGroot matrix construction
│   ├── agents.py          # Correlated AI advice (Kleinberg model) & hybrid belief updating
│   └── metrics.py         # Page diversity decomposition, Shannon entropy, & Monoculture Index
├── tests/
│   └── test_math.py       # Mathematical verification tests (Theorem residual < 1e-12)
├── experiments/           # Simulation parameter sweeps and scripts (upcoming)
├── requirements.txt       # Core dependencies
└── README.md
```

---

## Quickstart

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Mathematical Verification Tests
```bash
python tests/test_math.py
```

Expected output:
```text
✓ Page's Diversity Prediction Theorem verified to residual: 7.24e-14
✓ DeGroot trust matrix row-stochasticity verified (all rows sum to 1.0).
✓ Hybrid DeGroot boundary conditions (alpha=0.0 and alpha=1.0) verified.

ALL MATHEMATICAL INTEGRITY CHECKS PASSED SUCCESSFULLY.
```

---

## References

1. **Page, Scott E.** (2007). *The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies.* Princeton University Press.
2. **Kleinberg, Jon, & Raghavan, Manish.** (2021). *Algorithmic Monoculture and Separate Spaces.* ACM Conference on Economics and Computation (EC '21).
3. **Bommasani, Rishi, Creel, Kathleen A., et al.** (2024). *Generative Monoculture in Large Language Models.* arXiv:2407.08643.
4. **Watts, Duncan J., & Strogatz, Steven H.** (1998). *Collective dynamics of 'small-world' networks.* Nature, 393(6684), 440–442.
5. **Golub, Benjamin, & Jackson, Matthew O.** (2010). *Naïve Learning in Social Networks and the Wisdom of Crowds.* American Economic Journal: Microeconomics, 2(1), 112–149.

---

## License
MIT License. Open science and reproducible computational research.

