"""Check v5, task 3 addendum: robustness of q near z=-7 to solver nodes/tolerance and boundary handling (solve_c vs solve_c2)."""
import numpy as np
from centred import calibrated, solve_c
from centred2 import solve_c2
from feasible import x_at_theta
lines = []
for psi, rf in [(2.34, 0.10), (11.2, 0.15)]:
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0)
    for name, f in [("solve_c", solve_c), ("solve_c2", solve_c2)]:
        for n, tol in [(401, 1e-8), (801, 1e-8), (1601, 1e-8), (3201, 1e-8), (801, 1e-9), (801, 1e-6)]:
            try: sol = f(p, p, n=n, tol=tol, full=True)
            except Exception as e: lines.append(f"psi={psi} {name} n={n} tol={tol:g}: {e}"); continue
            z = np.linspace(-7, 7, 14001); w, wz = sol.sol(z); q = 1 + wz / w
            neg = q < 0
            lines.append(f"psi={psi} rf={rf} {name} n={n} tol={tol:g}: qmin={q.min():.6f} at z={z[np.argmin(q)]:.3f}; "
                         f"q<0 on {'[%.4f, %.4f]' % (z[neg].min(), z[neg].max()) if neg.any() else 'nowhere'}; q(-6.86)={np.interp(-6.86, z, q):.5f}")
s = "\n".join(lines); print(s); open("output/check_v5_qsign_robust.txt", "w").write(__doc__ + "\n" + s + "\n")
