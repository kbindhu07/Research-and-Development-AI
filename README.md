# Parametric Curve Fitting Tool


## 🧭 Overview

This repository contains a Python tool for **non-linear parametric curve fitting** using optimization techniques from the SciPy library.  
It estimates parameters of a custom 2D model that describes a trajectory `(x, y)` as a function of an inferred parameter `t`.

The script automatically:
- Loads coordinate data from a CSV file.
- Infers a pseudo-time variable from spatial geometry.
- Fits the parameters `(theta, M, X)` using non-linear least squares.
- Reports fit statistics and generates a comparison plot.

---

## ⚙️ Model Description

The parametric model is defined as:

\[
\begin{aligned}
x(t) &= t \cos(\theta) - e^{M|t|} \sin(0.3t)\sin(\theta) + X \\
y(t) &= 42 + t \sin(\theta) + e^{M|t|} \sin(0.3t)\cos(\theta)
\end{aligned}
\]

where:
- **θ (theta)**: orientation angle in radians  
- **M**: exponential modulation coefficient  
- **X**: horizontal offset parameter  

The parameter set `[θ, M, X]` is optimized by minimizing residuals between observed and modeled `(x, y)` values.

---

## 📦 Requirements

Ensure you have the following Python packages installed:

```bash
pip install numpy pandas scipy matplotlib
