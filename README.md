# Assignment for Research and Development

## 🧮 Fitted Model and Parameters

After parameter estimation using non-linear least squares, the following values were obtained:

\[
\theta = 0.826 \text{ rad}, \quad M = 0.0742, \quad X = 11.5793
\]

---

## 🧾 Final Model Equations (LaTeX Format)

\[
\begin{aligned}
x(t) &= t\cos(0.826) - e^{0.0742|t|}\sin(0.3t)\sin(0.826) + 11.5793, \\
y(t) &= 42 + t\sin(0.826) + e^{0.0742|t|}\sin(0.3t)\cos(0.826)
\end{aligned}
\]

---

## 🧩 Combined Representation (for Desmos)

\[
\left(t*\cos(0.826)-e^{0.0742\left|t\right|}\cdot\sin(0.3t)\sin(0.826)+11.5793,\ 
42+t*\sin(0.826)+e^{0.0742\left|t\right|}\cdot\sin(0.3t)\cos(0.826)\right)
\]

You can visualize this directly at:  
🔗 **[Desmos Graph Link](https://www.desmos.com/calculator/rfj91yrxob)**

---

## ➕ Additional Code / Math Used to Extract Variables

Parameter values were estimated using **non-linear least squares optimization** with `scipy.optimize.least_squares`.  
Below is a summary of the fitting method used:

```python
import numpy as np
from scipy.optimize import least_squares

def model(params, t):
    theta, M, X = params
    e = np.exp(M * np.abs(t))
    s03 = np.sin(0.3 * t)
    x = t * np.cos(theta) - e * s03 * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + e * s03 * np.cos(theta)
    return x, y

def residuals(params, t, x_obs, y_obs):
    x_pred, y_pred = model(params, t)
    return np.r_[x_pred - x_obs, y_pred - y_obs]

# Example parameter estimation
x0 = [0.8, 0.07, 11.5]  # initial guess
res = least_squares(residuals, x0, args=(t_data, x_data, y_data))
print(res.x)  # -> [0.826, 0.0742, 11.5793]

📘 Interpretation

θ (theta) controls the primary orientation of the curve.

M defines the rate of exponential modulation with 
∣
𝑡
∣
∣t∣.

X introduces a horizontal offset to align the curve with observed data.

The constant 42 represents a fixed vertical offset baseline.

🧪 Notes

The function combines linear and oscillatory components through sin(0.3t) and an exponential modulation term.

Parameter extraction was performed with high precision (xtol=1e-12, ftol=1e-12).

The results were validated visually using Desmos for confirmation.
