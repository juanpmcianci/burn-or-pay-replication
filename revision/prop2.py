"""Proposition 2 check: the lifecycle hurdle as a weighted average of local hurdles.

Linearising the CHT ODE around the baseline (omega = 0, kappa = 0) and dividing by W0 gives, for v = d log W,
    0.5 s^2 v'' + b(z) v' - k(z) v + dTheta(z) [kappa q(z) - h(z)] = 0,
    b = mu0 - s^2/2 + s^2 q,  k = alpha (1-alpha) n,  n = ((1-Theta0)/w)^alpha,  q = d log W0 / dz,
    h = c + L'(Theta0) + alpha (1-alpha) n / (1-Theta0)    (marginal cost of stake: rent + network).
So G_p = int m(z) dTheta_p(z) [kappa q - h] dz with m the discounted occupation density of the h-transformed
process, and kappa*_p = int pi_p(z) kstar(z) dz, kstar = h/q, pi_p ∝ m q dTheta_p.
dTheta_E = Theta'(z); dTheta_F = Theta'(z) F(z)/F(z_ref).
Boundary values (Dirichlet, Gordon) are perturbed exactly as in the solver.
Outputs: output/prop2_check.csv (formula vs finite-difference hurdles), output/prop2_weights.csv (weights, kstar(z)).
"""
import numpy as np, pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from multiprocessing import Pool
from centred import *
from feasible import x_at_theta
from cht import lock_cost

def pieces(p, x_lo=-7.0, x_hi=7.0, N=4001):
    sol = solve_c(p, p, x_lo=x_lo, x_hi=x_hi, full=True)
    z = np.linspace(x_lo, x_hi, N); w, wz = sol.sol(z)
    rho = rho_of(z, p); th = theta_of_rho(rho, p)
    e = 1e-6
    dth = (theta_of_rho(rho + e, p) - theta_of_rho(rho - e, p)) / (2 * e)
    Lp = ((lock_cost(th + e, p) - p.c * (th + e)) - (lock_cost(th - e, p) - p.c * (th - e))) / (2 * e)
    n = ((1 - th) / w) ** p.alpha
    q = 1 + wz / w
    h = p.c + Lp + p.alpha * (1 - p.alpha) * n / (1 - th)
    b = p.mu0 - p.sigma_A**2 / 2 + p.sigma_A**2 * q
    k = p.alpha * (1 - p.alpha) * n
    G = gamma(rho, th, p); En = p.E - p.w_B * fee(z, p)
    D = p.r_f - (p.mu0 - En + p.mu_S) - G                      # local Gordon denominator
    kcl = p.c + Lp + p.alpha * D / (1 - th)                       # local closed-form hurdle (Prop. 1 at z)
    return dict(z=z, w=w, q=q, h=h, b=b, k=k, dth=dth, F=fee(z, p), th=th, D=D, kcl=kcl)

def operator(P):
    z = P["z"]; dz = z[1] - z[0]; s2 = 0  # placeholder
    return dz

def solve_lin(P, src, bc, sig):
    """Solve 0.5 sig^2 v'' + b v' - k v + src = 0 with Dirichlet v = bc at both ends."""
    z = P["z"]; dz = z[1] - z[0]; N = len(z); b = P["b"]; k = P["k"]
    lo = 0.5 * sig**2 / dz**2 - b / (2 * dz); di = -sig**2 / dz**2 - k; up = 0.5 * sig**2 / dz**2 + b / (2 * dz)
    A = diags([lo[2:N-1], di[1:N-1], up[1:N-2]], [-1, 0, 1], format="csc")
    rhs = -src[1:N-1].copy()
    rhs[0] -= lo[1] * bc[0]; rhs[-1] -= up[N-2] * bc[1]
    v = np.empty(N); v[0], v[-1] = bc; v[1:N-1] = spsolve(A, rhs)
    return v, A

def weights(P, A, i0, sig):
    """Row i0 of (-A)^{-1}: discounted occupation density m(z) from z_ref."""
    N = len(P["z"]); dz = P["z"][1] - P["z"][0]
    e = np.zeros(N - 2); e[i0 - 1] = 1.0
    y = spsolve((-A).T.tocsc(), e)
    m = np.zeros(N); m[1:N-1] = y / dz
    return m

def job(a):
    case, psi, rf, kw = a
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, **{**dict(F_max=0.12, F_min=0.0), **kw})
    out = dict(case=case, psi=psi, rf=rf, **kw)
    try:
        x = x_at_theta(p); P = pieces(p); z = P["z"]; sig = p.sigma_A
        i0 = int(np.argmin(np.abs(z - x))); x0 = z[i0]
        Fr = fee(np.array([x0]), p)[0]
        M_end = 1 / (p.alpha * P["D"][[0, -1]])
        for kind, dT in [("E", P["dth"]), ("L", P["dth"] * P["F"] / Fr)]:
            vh, A = solve_lin(P, dT * P["h"], dT[[0, -1]] * M_end * P["kcl"][[0, -1]], sig)
            vq, _ = solve_lin(P, dT * P["q"], dT[[0, -1]] * M_end, sig)
            out[f"k{kind}_formula"] = vh[i0] / vq[i0]
            vh0, _ = solve_lin(P, dT * P["h"], np.zeros(2), sig)
            vq0, _ = solve_lin(P, dT * P["q"], np.zeros(2), sig)
            out[f"k{kind}_nobc"] = vh0[i0] / vq0[i0]
            out[f"G0{kind}_formula"] = -vh[i0]; out[f"S{kind}_formula"] = vq[i0]
            m = weights(P, A, i0, sig)
            pi = m * P["q"] * dT; pi = pi / np.trapezoid(pi, z)
            out[f"k{kind}_avg"] = np.trapezoid(pi * P["h"] / P["q"], z)
            out[f"mass{kind}_interior"] = np.trapezoid(m * P["q"] * dT, z) / vq0[i0]
            if kind == "E":
                out["intm"] = np.trapezoid(m, z); piE = pi
            else:
                piL = pi
        ks = P["h"] / P["q"]
        out["kstar_ref"] = ks[i0]; out["kcl_ref"] = P["kcl"][i0]; out["q_ref"] = P["q"][i0]
        out["qmin"] = P["q"].min()
        # monotonicity of kstar over the region carrying 99% of weight
        cdf = np.cumsum(piE) / np.sum(piE); sel = (cdf > 0.005) & (cdf < 0.995)
        out["kstar_monotone_99"] = bool(np.all(np.diff(ks[sel]) <= 1e-9))
        out["z_lo99"], out["z_hi99"] = z[sel][0] - x0, z[sel][-1] - x0
        # finite-difference hurdles (centred solver) for comparison
        out["kL_fd"] = hurdle(p, x, "F")[0]; out["kE_fd"] = hurdle(p, x, "U")[0]
        out["status"] = "ok"
        W = pd.DataFrame(dict(case=f"{case}|{psi}|{rf}|{kw}", dz=z - x0, piE=piE, piL=piL, kstar=ks, kcl=P["kcl"], q=P["q"], F=P["F"], th=P["th"]))
        return out, W
    except Exception as e:
        out["status"] = str(e)[:80]
        return out, None

if __name__ == "__main__":
    jobs = [("A", psi, rf, {}) for psi in [1.1, 1.5, 2.0] for rf in [0.10, 0.15, 0.20, 0.30]]
    jobs += [("B", 11.2, rf, {}) for rf in [0.15, 0.20, 0.30]]
    jobs += [("A", 1.5, 0.20, dict(s=s)) for s in [0.25, 1.5]] + [("B", 11.2, 0.20, dict(s=s)) for s in [0.25, 1.5]]
    with Pool(8) as pool:
        res = pool.map(job, jobs)
    d = pd.DataFrame([r[0] for r in res])
    d.to_csv("output/prop2_check.csv", index=False)
    pd.concat([r[1] for r in res if r[1] is not None]).to_csv("output/prop2_weights.csv", index=False)
    pd.set_option("display.width", 300)
    cols = ["case","psi","rf","s","kE_fd","kE_formula","kE_avg","kE_nobc","kL_fd","kL_formula","kL_avg","kL_nobc","kstar_ref","kcl_ref","q_ref","qmin","intm","kstar_monotone_99","z_lo99","z_hi99","status"]
    print(d.reindex(columns=cols).round(4).to_string(index=False))
