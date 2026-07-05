# AI R&D Assignment — Parametric Curve Fitting

## Result — Unknown variables

| Variable | Value |
|---|---|
| **θ (theta)** | 30° = 0.5235987756 rad |
| **M** | 0.03 |
| **X** | 55 |

**Fit quality:** RMS residual ≈ 3.5×10⁻⁶ over all 1500 points — i.e. an essentially exact fit (the tiny residual is just floating-point rounding from the 6-decimal precision in `xy_data.csv`).

## Desmos Link
https://www.desmos.com/calculator/juqw69elxn

### LaTeX equation

```
\left(t*\cos(0.5235987756)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987756)+55,42+t*\sin(0.5235987756)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987756)\right)
```
Domain: `6 ≤ t ≤ 60`

Equivalent, cleaner form using exact θ = 30°:

```
\left(t*\cos(30°)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(30°)+55,\ 42+t*\sin(30°)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(30°)\right)
```

## Approach

The given curve is:
```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

The dataset gives 1500 `(x, y)` points, but **not** the `t` value that produced each point — so naively this looks like a (1500 + 3)-parameter fitting problem (one hidden `t` per point, plus θ, M, X).

**Key simplification.** Define `u = e^(M|t|)·sin(0.3t)`. Then the equations are exactly a 2D **rotation by θ** applied to the vector `(t, u)`, followed by a translation by `(X, 42)`:

```
[x − X, y − 42]ᵀ = R(θ) · [t, u]ᵀ
```

Because a rotation matrix is orthogonal (`R(θ)⁻¹ = R(θ)ᵀ = R(−θ)`), for **any** candidate `(θ, X)` we can invert this in closed form and recover what `t` and `u` *would have to be* for every data point:

```
t_i =  (x_i − X)·cos(θ) + (y_i − 42)·sin(θ)
u_i = −(x_i − X)·sin(θ) + (y_i − 42)·cos(θ)
```

If `(θ, X)` are correct, then `u_i` must equal `e^(M|t_i|)·sin(0.3·t_i)` for one consistent `M`, for every point simultaneously. This collapses the problem to a plain **3-parameter nonlinear least-squares fit**:

```
residual_i(θ, M, X) = u_i − e^(M|t_i|)·sin(0.3·t_i)
```

minimize `Σ residual_i²` over `θ ∈ (0°, 50°)`, `M ∈ (−0.05, 0.05)`, `X ∈ (0, 100)`.

### Numerical solution steps (`solution.py`)
1. Load `xy_data.csv` file with 1500 points.
2. Do global optimization for 3D parameter space within bounds using `scipy.optimize.differential_evolution` (in order not to be stuck in local minimum because of oscillatory terms like `sin(0.3t)` and `sin/cos(θ)`).
3. Do local optimization for best obtained point using `scipy.optimize.least_squares` (Levenberg-Marquardt-like algorithm) in order to obtain very accurate results.
4. Sanity check: all obtained `t_i`'s belong to the range `(6, 60)` as stated in the problem – which means that global minimum is indeed obtained.
5. Visualization check: draw a curve and overlay it with points (`fit_overlay.png`) – the curve perfectly fits all the points.

### Why this beats brute-force / ICP approaches
The naive approach to solving this (such as taking t at each point to be an independent variable or performing nearest neighbor "distance from point to curve" fitting) would require optimization in greater than 1500 dimensions. The smart algorithm mentioned above succeeds in analytically eliminating all 1500 nuisance parameters t, leaving us with a simple 3D least squares problem, which is quick to solve.

## Files
- `flam_solution.py` - full fitting script 
- `xy_data.csv` - original data 
- `fit_overlay.png` - verification plot: fitted curve overlaid on given data points
- `Output of flam_solution.py` - output values for flam_solution.py


## How to run
```bash
pip install numpy pandas scipy matplotlib
python3 solution.py
```
