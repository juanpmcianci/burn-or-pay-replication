"""Check v6: rescue attempts for (psi, r_f) = (2.34, 0.10), omega0 = 0.9, E = 0.057 (failed on [-12,7], 1087 nodes).
Reports end-point D and R, then tries solve_c2 baseline and the hurdles on smaller domains / initial-guess scalings."""
import numpy as np
from centred import calibrated, fee, rho_of, theta_of_rho, gamma
from centred2 import gap2, solve_c2
from feasible import x_at_theta
p = calibrated(beta=2.34, kappa=0.0, r_f=0.10, F_max=0.12, F_min=0.0, s=0.75, w_B=0.9, E=0.057)
x = x_at_theta(p); lines = [__doc__, f"z0={x:.5f}"]
for z in (-12, -9, -7, -5, -3, 0, x, 7):
    zz = np.array([float(z)]); th = theta_of_rho(rho_of(zz, p), p); G = gamma(rho_of(zz, p), th, p)
    R = p.r_f - p.mu_S + p.E - p.w_B * fee(zz, p) - G; lines.append(f"z={z:7.3f} F={fee(zz,p)[0]:.4f} Theta={th[0]:.4f} R={R[0]:.5f} D=R-mu0={R[0]-p.mu0:.5f}")
def hur(kind, **kw):
    g0 = gap2(p, x, kind, 0.0, **kw); g2 = gap2(p, x, kind, 0.2, **kw); return -g0 / ((g2 - g0) / 0.2)
for xlo, n in [(-12, 1087), (-9, 915), (-7, 801), (-5, 687), (-12, 2173)]:
    for guess in (1.0, 0.5, 2.0):
        try:
            sol = solve_c2(p, p, x_lo=xlo, n=n, full=True, guess=guess); w, wz = sol.sol(np.array([x]))
            kE = hur("U", x_lo=xlo, n=n); kL = hur("F", x_lo=xlo, n=n)
            lines.append(f"x_lo={xlo} n={n} guess={guess}: q={1+wz[0]/w[0]:.5f} kE={kE:.5f} kL={kL:.5f} ratio={kL/kE:.5f}")
            break
        except Exception as e:
            lines.append(f"x_lo={xlo} n={n} guess={guess}: FAIL {str(e)[:60]}")
s = "\n".join(lines); print(s); open("output/check_v6_omega09_rf10.txt", "w").write(s + "\n")
