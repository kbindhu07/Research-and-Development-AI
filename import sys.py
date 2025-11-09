import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def load_xy(path):
    df = pd.read_csv(path)
    if not {'x','y'}.issubset(set(df.columns.str.lower())):
        raise RuntimeError("CSV must have columns named 'x' and 'y'.")
    x = df['x'].values if 'x' in df.columns else df['X'].values
    y = df['y'].values if 'y' in df.columns else df['Y'].values
    return x.astype(float), y.astype(float)

def model(params, t):
    theta, M, X = params
    et = np.exp(M * np.abs(t))
    s03 = np.sin(0.3 * t)
    x = t * np.cos(theta) - et * s03 * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + et * s03 * np.cos(theta)
    return x, y

def residuals(params, t, x_obs, y_obs):
    x_pred, y_pred = model(params, t)
    return np.r_[x_pred - x_obs, y_pred - y_obs]

def initial_guess_from_data(x, y):
    XY = np.column_stack([x, y])
    XYc = XY - XY.mean(axis=0)
    _, svals, vh = np.linalg.svd(XYc, full_matrices=False)
    principal = vh[0]  
    theta0 = np.arctan2(principal[1], principal[0])
    theta0 = float(np.clip(theta0, np.deg2rad(1e-6), np.deg2rad(50.0)))
    u = np.array([np.cos(theta0), np.sin(theta0)])
    s_raw = x * u[0] + y * u[1]
    s_min, s_max = s_raw.min(), s_raw.max()
    C = s_max - 60.0
    X0 = (C - 42.0 * np.sin(theta0)) / np.cos(theta0)
    M0 = 0.0
    return np.array([theta0, M0, float(np.clip(X0, 0.001, 99.999))])

def fit(x, y):
    def infer_t(params):
        theta, M, X = params
        u = np.array([np.cos(theta), np.sin(theta)])
        # p - origin
        sx = x - X
        sy = y - 42.0
        return sx * u[0] + sy * u[1]

    def fun(params):
        t_est = infer_t(params)
        return residuals(params, t_est, x, y)

    theta_bounds = (np.deg2rad(1e-6), np.deg2rad(50.0))
    bounds = ([theta_bounds[0], -0.05, 0.0],
              [theta_bounds[1],  0.05, 100.0])
    x0 = initial_guess_from_data(x, y)
    res = least_squares(fun, x0, bounds=bounds, xtol=1e-12, ftol=1e-12, verbose=2)
    return res

def compute_l1(params, x, y):
    theta, M, X = params
    u = np.array([np.cos(theta), np.sin(theta)])
    sx = x - X
    sy = y - 42.0
    t_est = sx * u[0] + sy * u[1]
    x_pred, y_pred = model(params, t_est)
    return np.mean(np.abs(x_pred - x) + np.abs(y_pred - y))

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/fit_curve.py path/to/xy_data.csv")
        sys.exit(1)
    path = sys.argv[1]
    x, y = load_xy(path)
    res = fit(x, y)
    theta, M, X = res.x
    print(f"theta (rad) = {theta:.8f}, theta (deg) = {np.rad2deg(theta):.6f}")
    print(f"M = {M:.8e}")
    print(f"X = {X:.8f}")
    l1 = compute_l1(res.x, x, y)
    print(f"L1 (mean abs x+y diff on data) = {l1:.6e}")
    u = np.array([np.cos(theta), np.sin(theta)])
    t_est = (x - X) * u[0] + (y - 42.0) * u[1]
    xp, yp = model(res.x, t_est)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=6, label='data')
    plt.scatter(xp, yp, s=6, label='model', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('fit_result.png', dpi=150)
    print("plot saved to fit_result.png")

if __name__ == '__main__':
    main()