"""Check v6: is kappa*_omega > kappa*_E built in by the baseline omega0 = 0 and F_min = 0?

Solver: centred2.gap2 (paper's Table 2 solver) on [-12, 7], 1087 nodes, tol 1e-8. Hurdle = -0.2 G(0)/(G(0.2)-G(0)),
kind "U" (issuance) and "F" (burn share). gap2 builds the baseline economy as with_(p, kappa=0), so setting
p.w_B = omega0 makes both the baseline (Theta_0 path, W0) and the derivative centre equal omega0:
  "F" gap = -(1/F(z0)) d log W(z0)/d w_B, central difference w_B = omega0 +- 0.002;
  "U" gap = d log W(z0)/dE, central difference E +- 2e-4, at w_B = omega0.
Model: rho = E + (1-w_B) F,  E_net = E - w_B F (enters the Gordon denominator), F = F_min + (F_max-F_min)/(1+e^{z/s}).

Run 1 (baseline burn share omega0 in {0.5, 0.9}). Held fixed: F_max=0.12, F_min=0, s=0.75, x0=0, c=0.01, mu0=0.02,
  alpha=0.3, sigma_A=0.35, u = u_for(0.45, 0.06, psi, c) (unchanged). E = 0.06 - (1-omega0) F_ref, F_ref = 0.03.
  Then at z0 = s ln 3 (same as Table 2): F = 0.03, rho = 0.06, Theta = 0.45, and E_net = E - omega0 F_ref = 0.03,
  so the reward ratio, the staking ratio, the fee yield, the net issuance and the closed-form kappa* at the reference
  state are all exactly those of Table 2. What changes: away from z0, rho moves by (1-omega0) dF (flatter Theta path),
  E_net moves by -omega0 dF (deflation in high-fee states).
Run 2 (fee floor F_min in {0.01, 0.02}, omega0 = 0, F_max = 0.12). Two ways to keep rho = 0.06, Theta = 0.45 at z0:
  'E': E = 0.03, so F(z0) = 0.03 and z0 shifts right to s ln((F_max-F_min)/(0.03-F_min) - 1);
  'z': z0 = s ln 3 kept, F(z0) = F_min + (F_max-F_min)/4, E = 0.06 - F(z0).
Closed form kappa* = rho/Theta + (Gamma + alpha D)/(1-Theta), D = r_f - mu0 + E_net(z0) - Gamma (calibB.closed_form).
Elasticity eps = 1 + (d log F/dz)/q, q = d log W0/dz = 1 + w'/w from the baseline solve_c2 solution at z0.
Prop. 2 decomposition (logic of prop2.py, with solve_c2 on [-12,7] in place of solve_c on [-7,7]): linearised ODE,
weights pi_E ∝ m q Theta_rho, local hurdle khat = h/q; reports kE_formula, kL_formula, Cov_piE(F, khat),
Cov/E[F] (should equal kL - kE), d khat/dz at z0, and whether khat is non-increasing on the 99% pi_E mass."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from centred import calibrated, fee, rho_of, theta_of_rho, gamma
from centred2 import gap2, solve_c2
from feasible import x_at_theta
from calibB import closed_form
from cht import lock_cost

XLO, XHI, NN = -12.0, 7.0, 1087
ROWS = [(1.5, 0.20), (2.34, 0.20), (11.2, 0.20), (2.34, 0.10)]

def variants():
    V = [("base", dict(w_B=0.0, E=0.03, F_min=0.0))]
    for w in (0.5, 0.9):
        V.append((f"omega0={w}", dict(w_B=w, E=0.06 - (1 - w) * 0.03, F_min=0.0)))
    for fm in (0.01, 0.02):
        V.append((f"Fmin={fm}|E", dict(w_B=0.0, E=0.03, F_min=fm)))
        Fz = fm + (0.12 - fm) / 4
        V.append((f"Fmin={fm}|z", dict(w_B=0.0, E=0.06 - Fz, F_min=fm)))
    return V

def hur(p, x, kind):
    g0 = gap2(p, x, kind, 0.0, x_lo=XLO, n=NN); g2 = gap2(p, x, kind, 0.2, x_lo=XLO, n=NN)
    return -g0 / ((g2 - g0) / 0.2), g0

def lin_pieces(p, N=7601):
    sol = solve_c2(p, p, x_lo=XLO, x_hi=XHI, n=NN, full=True)
    z = np.linspace(XLO, XHI, N); w, wz = sol.sol(z)
    rho = rho_of(z, p); th = theta_of_rho(rho, p); e = 1e-6
    dth = (theta_of_rho(rho + e, p) - theta_of_rho(rho - e, p)) / (2 * e)
    Lp = ((lock_cost(th + e, p) - p.c * (th + e)) - (lock_cost(th - e, p) - p.c * (th - e))) / (2 * e)
    n = ((1 - th) / w) ** p.alpha; q = 1 + wz / w
    h = p.c + Lp + p.alpha * (1 - p.alpha) * n / (1 - th)
    b = p.mu0 - p.sigma_A**2 / 2 + p.sigma_A**2 * q; k = p.alpha * (1 - p.alpha) * n
    G = gamma(rho, th, p); En = p.E - p.w_B * fee(z, p)
    D = p.r_f - (p.mu0 - En + p.mu_S) - G
    kcl = p.c + Lp + p.alpha * D / (1 - th)
    return dict(z=z, w=w, q=q, h=h, b=b, k=k, dth=dth, F=fee(z, p), th=th, D=D, kcl=kcl)

def solve_lin(P, src, bc, sig):
    z = P["z"]; dz = z[1] - z[0]; N = len(z); b = P["b"]; k = P["k"]
    lo = 0.5 * sig**2 / dz**2 - b / (2 * dz); di = -sig**2 / dz**2 - k; up = 0.5 * sig**2 / dz**2 + b / (2 * dz)
    A = diags([lo[2:N-1], di[1:N-1], up[1:N-2]], [-1, 0, 1], format="csc")
    rhs = -src[1:N-1].copy(); rhs[0] -= lo[1] * bc[0]; rhs[-1] -= up[N-2] * bc[1]
    v = np.empty(N); v[0], v[-1] = bc; v[1:N-1] = spsolve(A, rhs)
    return v, A

def occupation(P, A, i0):
    N = len(P["z"]); dz = P["z"][1] - P["z"][0]
    e = np.zeros(N - 2); e[i0 - 1] = 1.0
    m = np.zeros(N); m[1:N-1] = spsolve((-A).T.tocsc(), e) / dz
    return m

def prop2_block(p, x):
    P = lin_pieces(p); z = P["z"]; sig = p.sigma_A
    i0 = int(np.argmin(np.abs(z - x))); Fr = fee(np.array([z[i0]]), p)[0]
    out = dict(qmin=P["q"].min(), D_lo=P["D"][0])
    # Gordon-perturbation Dirichlet ends where D > 0 (as prop2.py); zero otherwise (left end is 12 units away)
    Dend = P["D"][[0, -1]]; M_end = np.where(Dend > 0, 1 / (p.alpha * np.maximum(Dend, 1e-12)), 0.0)
    for kind, dT in [("E", P["dth"]), ("L", P["dth"] * P["F"] / Fr)]:
        vh, A = solve_lin(P, dT * P["h"], dT[[0, -1]] * M_end * P["kcl"][[0, -1]], sig)
        vq, _ = solve_lin(P, dT * P["q"], dT[[0, -1]] * M_end, sig)
        out[f"k{kind}_formula"] = vh[i0] / vq[i0]
        if kind == "E":
            m = occupation(P, A, i0)
            piE = m * P["q"] * P["dth"]; piE = piE / np.trapezoid(piE, z)
    kh = P["h"] / P["q"]; F = P["F"]
    EF = np.trapezoid(piE * F, z); Ek = np.trapezoid(piE * kh, z); EFk = np.trapezoid(piE * F * kh, z)
    out.update(kE_avg=Ek, kL_avg=EFk / EF, cov=EFk - EF * Ek, cov_over_EF=(EFk - EF * Ek) / EF,
               corr=(EFk - EF * Ek) / np.sqrt(np.trapezoid(piE * (F - EF)**2, z) * np.trapezoid(piE * (kh - Ek)**2, z)))
    dk = np.gradient(kh, z); out["dkhat_dz_ref"] = dk[i0]
    j = np.abs(z - z[i0]) <= 0.5; out["dkhat_dz_max_pm0.5"] = dk[j].max()
    cdf = np.cumsum(piE) / np.sum(piE); sel = (cdf > 0.005) & (cdf < 0.995)
    out["khat_noninc_99"] = bool(np.all(np.diff(kh[sel]) <= 1e-9)); out["frac_mass_dk>0"] = np.trapezoid(piE * (dk > 0), z)
    out["khat_ref"] = kh[i0]; out["meanpiE_dz"] = np.trapezoid(piE * z, z) - z[i0]
    return out

def job(a):
    psi, rf, name, kw = a
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, s=0.75, **kw)
    out = dict(psi=psi, rf=rf, variant=name, w_B=p.w_B, E=p.E, F_min=p.F_min)
    try:
        x = x_at_theta(p); Fx = fee(np.array([x]), p)[0]
        out.update(z0=x, F_ref=Fx, rho_ref=rho_of(np.array([x]), p)[0],
                   Theta_ref=theta_of_rho(rho_of(np.array([x]), p), p)[0], Enet_ref=p.E - p.w_B * Fx)
        out["kstar_cf"] = closed_form(psi, rf, E=out["Enet_ref"])
        sol = solve_c2(p, p, x_lo=XLO, n=NN, full=True); w, wz = sol.sol(np.array([x]))
        q = 1 + wz[0] / w[0]; hh = 1e-4
        dlF = (np.log(fee(np.array([x + hh]), p)) - np.log(fee(np.array([x - hh]), p)))[0] / (2 * hh)
        out.update(q=q, dlogF_dz=dlF, eps=1 + dlF / q)
        out["kE"], out["G0E"] = hur(p, x, "U"); out["kL"], out["G0L"] = hur(p, x, "F"); out["status"] = "ok"
    except Exception as e:
        out["status"] = "hurdle: " + str(e)[:60]; return out
    try:
        out.update(prop2_block(p, x))
    except Exception as e:
        out["p2_status"] = str(e)[:60]
    return out

if __name__ == "__main__":
    jobs = [(psi, rf, n, kw) for psi, rf in ROWS for n, kw in variants()]
    with Pool(12) as pool: d = pd.DataFrame(pool.map(job, jobs))
    d["ratio"] = d.kL / d.kE; d["kL_minus_kE"] = d.kL - d.kE
    d.to_csv("output/check_v6_baseline.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
    f = lambda v: f"{v:.5f}"
    t1 = d[["psi", "rf", "variant", "w_B", "E", "F_min", "z0", "F_ref", "rho_ref", "Theta_ref", "Enet_ref", "status"]]
    t2 = d[["psi", "rf", "variant", "kstar_cf", "kE", "kL", "ratio", "q", "dlogF_dz", "eps"]]
    t3 = d.reindex(columns=["psi", "rf", "variant", "kE_formula", "kL_formula", "kE_avg", "kL_avg", "kL_minus_kE", "cov",
                            "cov_over_EF", "corr", "dkhat_dz_ref", "dkhat_dz_max_pm0.5", "khat_noninc_99", "frac_mass_dk>0",
                            "qmin", "D_lo", "meanpiE_dz", "p2_status"])
    s = "\n== reference state ==\n" + t1.to_string(index=False, float_format=f) + \
        "\n\n== hurdles and elasticity ==\n" + t2.to_string(index=False, float_format=f) + \
        "\n\n== Prop. 2 decomposition ==\n" + t3.to_string(index=False, float_format=f)
    print(s); open("output/check_v6_baseline.txt", "w").write(__doc__ + s + "\n")
