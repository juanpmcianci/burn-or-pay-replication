"""Check v5, task 3: where q(z) = d log W0/dz < 0 (psi=2.34, r_f=0.10; also calibration B r_f=0.15), and the share of
nu_E q and nu_omega q there. Uses prop2.pieces / weights (centred.solve_c, Gordon at both ends; D0 > 0 at the lower end
here, so this is the same boundary condition as centred2). Grid spacing 14/4000 as in prop2; domains [-7,7], [-9,7], [-12,7].
nu_E = g Theta_rho, nu_omega = nu_E F/F(z0); g = row i0 of the inverse of the discretised operator (prop2.weights)."""
import numpy as np, pandas as pd
import prop2
from centred import calibrated, fee
from feasible import x_at_theta

rows, lines = [], []
for psi, rf in [(2.34, 0.10), (11.2, 0.15)]:
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0)
    x = x_at_theta(p)
    for xlo in (-7.0, -9.0, -12.0):
        N = int(round((7 - xlo) / (14 / 4000))) + 1
        Pp = prop2.pieces(p, x_lo=xlo, x_hi=7.0, N=N); z = Pp["z"]; q = Pp["q"]; sig = p.sigma_A
        i0 = int(np.argmin(np.abs(z - x))); Fr = fee(np.array([z[i0]]), p)[0]
        _, A = prop2.solve_lin(Pp, Pp["dth"] * Pp["h"], np.zeros(2), sig)
        m = prop2.weights(Pp, A, i0, sig)
        neg = q < 0
        out = dict(psi=psi, rf=rf, x_lo=xlo, N=N, qmin=q.min(), z_at_qmin=z[np.argmin(q)], q_ref=q[i0])
        if neg.any():
            out["neg_lo"], out["neg_hi"] = z[neg].min(), z[neg].max()
            out["n_neg_nodes"] = int(neg.sum()); out["contiguous"] = bool(np.all(np.diff(np.where(neg)[0]) == 1))
        for kind, nu in [("E", m * Pp["dth"]), ("om", m * Pp["dth"] * Pp["F"] / Fr)]:
            f = nu * q; tot = np.trapezoid(f, z); totabs = np.trapezoid(np.abs(f), z)
            fn = np.where(neg, f, 0.0)
            out[f"share_{kind}_signed"] = np.trapezoid(fn, z) / tot
            out[f"share_{kind}_abs"] = np.trapezoid(np.abs(fn), z) / totabs
            out[f"nu_{kind}_mass_share"] = np.trapezoid(np.where(neg, nu, 0), z) / np.trapezoid(nu, z)
        rows.append(out)
        if xlo == -7.0 and psi == 2.34:
            sel = z < -6.0
            lines.append("psi=2.34 rf=0.10 [-7,7]: q on z in [-7,-6] (every 20th node)\n" +
                         pd.DataFrame(dict(z=z[sel][::20], q=q[sel][::20])).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
d = pd.DataFrame(rows); pd.set_option("display.width", 300)
s = d.T.to_string(float_format=lambda v: f"{v:.6g}") + "\n\n" + "\n".join(lines)
print(s); open("output/check_v5_qsign.txt", "w").write(__doc__ + "\n" + s + "\n")
