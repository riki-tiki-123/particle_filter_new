# Sensor Fusion – Particle Filter for Ball Tracking  
**Portfolio Exam 2 – THWS Würzburg-Schweinfurt**  
**Course: Reasoning and Decision Making under Uncertainty (SS2025)**  
**Author: [Your Name]**  
**Due: 29.06.2025**

---

## 🔍 Objective

This project implements a **Particle Filter** (also known as the **Condensation algorithm**) to estimate the trajectories of one or more flying balls from **noisy, intermittent observations**. The filter estimates both **positions and velocities** over time, despite:
- Unknown initial conditions
- High observation uncertainty
- Sensor dropouts
- Indistinguishability of objects

This solution generalizes the Kalman filter to non-linear, non-Gaussian cases using a **sequential Monte Carlo (SMC)** approach. It was built as part of Portfolio Exam 2 for the module "Reasoning and Decision Making under Uncertainty".

---

## 📦 File Overview

| File              | Description |
|------------------|-------------|
| `run.py`          | Runs the full simulation, observation generation, particle filter loop, and result plotting |
| `simulator.py`    | Simulates projectile motion under gravity and generates noisy, dropped-out observations |
| `particle_filter.py` | Contains the core Particle Filter logic, including prediction, update, and resampling |
| `clustering.py`   | Implements weighted KMeans (and optionally KDE) to extract multi-modal state estimates from particles |
| `utils.py`        | Common utility functions including particle initialization and normalization |
| `README.md`       | This documentation file |

---

## 🧠 Algorithm: Condensation (Particle Filter)

This implementation follows the **Condensation algorithm** as an alternative to the Kalman Filter for non-linear systems. Key features include:

- **State Representation**: Each particle represents a hypothesis of the full state vector  
  \[
  x = [x_1, y_1, v_{x1}, v_{y1}, ..., x_n, y_n, v_{xn}, v_{yn}]
  \]

- **Transition Model**: Applies gravity, motion integration, and Gaussian process noise  
  Matches the real-world physics of projectile motion + collisions.

- **Measurement Model**: Observations are noisy 2D positions of the balls  
  Likelihoods are computed using Mahalanobis distance across all permutations (because balls are indistinguishable).

- **Resampling**: Systematic resampling ensures high-weight particles are propagated  
  Prevents particle depletion.

- **Estimation**: Clustering (KMeans) extracts **multi-modal position estimates**  
  Needed since the filter tracks **multiple indistinguishable** objects.

---

## 🧪 Features & Design Choices

### ✅ **Simulation (`simulator.py`)**
- Ball motion under gravity (no air drag)
- Realistic sensor:
  - Noisy Gaussian observations
  - Dropout simulation (random frames with no data)
  - Non-uniform observation intervals

### ✅ **Particle Filter (`particle_filter.py`)**
- Fully vectorized propagation using numpy
- Elastic collision resolution (between balls)
- Permutation-aware likelihood calculation (for indistinguishable observations)

### ✅ **Clustering (`clustering.py`)**
- Extracts most probable positions from particles using weighted KMeans
- KDE alternative also included for robustness

### ✅ **Execution Script (`run.py`)**
- Easily change parameters: number of balls, noise levels, dropout rates, etc.
- Plots true vs estimated ball positions over time
- Saves figure for later discussion/interview

---

## 📈 Visualization

The final output includes a figure:

