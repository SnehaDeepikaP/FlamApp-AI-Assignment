"""
APPROACH
--------
Note the equations are a 2D rotation by theta applied to the vector (t, u),
where u = e^(M|t|) * sin(0.3t), followed by a translation by (X, 42):

    [x - X, y - 42]^T = R(theta) @ [t, u]^T

Because R(theta) is orthogonal (R^-1 = R^T), for ANY candidate (theta, X) we can
invert this in closed form to recover, for every data point, what "t" and "u"
would have to be:

    t_i =  (x_i - X)*cos(theta) + (y_i - 42)*sin(theta)
    u_i = -(x_i - X)*sin(theta) + (y_i - 42)*cos(theta)

If (theta, X) are correct, u_i must equal e^(M|t_i|) * sin(0.3*t_i) for a
single, consistent M across all points. This collapses what looks like a
(1500 + 3)-parameter fitting problem (each point has its own hidden t) into a
plain 3-parameter nonlinear least-squares problem:

    residual_i(theta, M, X) = u_i - e^(M|t_i|) * sin(0.3*t_i)

We minimize sum(residual_i^2) over (theta, M, X) inside the given bounds.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution

df = pd.read_csv('xy_data.csv')
x = df['x'].values
y = df['y'].values

def residuals(params):
    theta, M, X = params
    t = (x - X) * np.cos(theta) + (y - 42) * np.sin(theta)
    u = -(x - X) * np.sin(theta) + (y - 42) * np.cos(theta)
    u_hat = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return u - u_hat

def cost(params):
    return np.sum(residuals(params) ** 2)

bounds = [(1e-3, np.deg2rad(50) - 1e-3),   # theta
          (-0.05 + 1e-4, 0.05 - 1e-4),      # M
          (1e-3, 100 - 1e-3)]               # X

de_result = differential_evolution(
    cost, bounds, seed=42, maxiter=300, popsize=40,
    tol=1e-14, polish=True, workers=1
)

lo = [b[0] for b in bounds]
hi = [b[1] for b in bounds]
res = least_squares(residuals, de_result.x, bounds=(lo, hi),
                     xtol=1e-15, ftol=1e-15, gtol=1e-15)

theta, M, X = res.x
final_cost = np.sum(res.fun ** 2)

print("=== FIT RESULTS ===")
print(f"theta = {theta:.10f} rad  = {np.rad2deg(theta):.6f} deg")
print(f"M     = {M:.10f}")
print(f"X     = {X:.10f}")
print(f"Sum of squared residuals = {final_cost:.3e}  (over {len(x)} points)")
print(f"RMS residual             = {np.sqrt(final_cost/len(x)):.3e}")

t_recovered = (x - X) * np.cos(theta) + (y - 42) * np.sin(theta)
print(f"Recovered t range: [{t_recovered.min():.4f}, {t_recovered.max():.4f}]  (expected within (6, 60))")

theta_r, M_r, X_r = np.deg2rad(30), 0.03, 55.0
print("\n=== CHECK AGAINST CLEAN VALUES theta=30deg, M=0.03, X=55 ===")
print("Cost at clean values:", cost([theta_r, M_r, X_r]))