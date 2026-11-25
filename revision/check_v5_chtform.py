"""Check v5, task 4: hurdles under CHT's constant-centre feedback mu_A = mu0 + kappa (Theta - theta_bar), theta_bar = 0.45.
The baseline economy now depends on kappa, so G(kappa) is not affine; the hurdle is the root of G(kappa) = 0 (brentq),
with both perturbed solves (w_B +- 2e-3, or E +- 2e-4) at the same kappa. Solver: a copy of centred2.solve_c2 with the
feedback centre as an argument (constant 0.45 here; the baseline path Theta_0(z) reproduces the centred form).
Boundary: Gordon value at the lower end if D0 > 0, power-law Robin condition if D0 <= 0 < R0 (as centred2)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from centred import calibrated, theta_of_rho, rho_of, gamma, fee, with_
from centred2 import NoPrice
from feasible import x_at_theta

def solve_gen(p, centre, x_lo=-7.0, x_hi=7.0, n=801, tol=1e-8):
    s2 = p.sigma_A ** 2
    def parts(x):
        th = theta_of_rho(rho_of(x, p), p)
        mA = p.mu0 + p.kappa * (th - centre(x))
        G = gamma(rho_of(x, p), th, p) + p.D
        En = p.E - p.w_B * fee(x, p)
        return mA, G, En, th, p.r_f - p.mu_S + En - G
    def rhs(x, y):
        w, wx = y; w = np.maximum(w, 1e-12); mA, G, En, tn, R = parts(x)
        flow = (mA - R) * w + (1 - p.alpha) * (1 - tn) ** p.alpha * w ** (1 - p.alpha)
        return np.vstack([wx, -(2.0 / s2) * ((mA + 0.5 * s2) * wx + flow)])
    xs = np.linspace(x_lo, x_hi, n); mA, G, En, th, R = parts(xs); den = R - mA
    if den[-1] <= 0: raise NoPrice(f"upper-end D {den[-1]:.4f}")
    if R[0] <= 0 and den[0] <= 0: raise NoPrice(f"lower-end R {R[0]:.4f}")
    gord = lambda i: (1 - th[i]) * ((1 - p.alpha) / den[i]) ** (1 / p.alpha)
    left_gordon = den[0] > 0
    if not left_gordon:
        b = mA[0] - s2 / 2; g = (-b + np.sqrt(b * b + 2 * s2 * R[0])) / s2
    w0 = np.empty(n); ok = den > 0
    w0[ok] = (1 - th[ok]) * ((1 - p.alpha) / den[ok]) ** (1 / p.alpha)
    if not ok.all():
        j = np.where(~ok)[0].max() + 1
        gg = g if not left_gordon else 0.5
        w0[:j] = w0[j] * np.exp((gg - 1) * (xs[:j] - xs[j]))
    def bc(a, bb):
        left = a[0] - gord(0) if left_gordon else a[1] - (g - 1) * a[0]
        return np.array([left, bb[0] - gord(n - 1)])
    sol = solve_bvp(rhs, bc, xs, np.vstack([w0, np.gradient(w0, xs)]), tol=tol, max_nodes=1000000)
    if not sol.success: raise RuntimeError(sol.message)
    if np.any(sol.sol(xs)[0] <= 0): raise RuntimeError("non-positive w")
    return lambda z: np.log(sol.sol(np.atleast_1d(z))[0]) + np.atleast_1d(z), den[0]

DW, H = 2e-3, 2e-4
def gap_gen(p, x, kind, kappa, centre, **kw):
    q = with_(p, kappa=kappa)
    if kind == "F":
        up = solve_gen(with_(q, w_B=DW), centre, **kw)[0](x); dn = solve_gen(with_(q, w_B=-DW), centre, **kw)[0](x)
        return (-(up - dn) / (2 * DW) / fee(np.atleast_1d(x), q))[0]
    up = solve_gen(with_(q, E=q.E + H), centre, **kw)[0](x); dn = solve_gen(with_(q, E=q.E - H), centre, **kw)[0](x)
    return ((up - dn) / (2 * H))[0]

def P(psi, rf): return calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75)
CONST = lambda z: 0.45

def scan(a):
    psi, rf, xlo, kind, k = a
    p = P(psi, rf); x = x_at_theta(p); n = int(round((7 - xlo) / (14 / 800))) + 1
    try:
        D0 = solve_gen(with_(p, kappa=k), CONST, x_lo=xlo, n=n)[1]
        return dict(psi=psi, rf=rf, x_lo=xlo, kind=kind, kappa=k, D0=D0, G=gap_gen(p, x, kind, k, CONST, x_lo=xlo, n=n))
    except Exception as e:
        return dict(psi=psi, rf=rf, x_lo=xlo, kind=kind, kappa=k, err=str(e)[:50])

def root(a):
    psi, rf, xlo, kind, form = a
    p = P(psi, rf); x = x_at_theta(p); n = int(round((7 - xlo) / (14 / 800))) + 1
    centre = CONST if form == "const" else (lambda z: theta_of_rho(rho_of(z, p), p))
    f = lambda k: gap_gen(p, x, kind, k, centre, x_lo=xlo, n=n)
    try:
        if form == "centred":  # affine: two-point, as the paper
            g0, g2 = f(0.0), f(0.2); return dict(psi=psi, rf=rf, x_lo=xlo, kind=kind, form=form, kstar=-0.2 * g0 / (g2 - g0))
        ks = brentq(f, 0.0, 0.6, xtol=1e-7)
        h = 1e-3; slope = (f(ks + h) - f(ks - h)) / (2 * h)
        return dict(psi=psi, rf=rf, x_lo=xlo, kind=kind, form=form, kstar=ks, dG_dkappa_at_root=slope, G0=f(0.0))
    except Exception as e:
        return dict(psi=psi, rf=rf, x_lo=xlo, kind=kind, form=form, err=str(e)[:60])

if __name__ == "__main__":
    cases = [(1.5, 0.20), (2.34, 0.20)]
    rjobs = [(psi, rf, xlo, kind, form) for psi, rf in cases for xlo in (-7.0, -9.0, -12.0) for kind in "UF" for form in ("const", "centred")]
    kgrid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.66, 0.67, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0]
    sjobs = [(psi, rf, xlo, kind, k) for psi, rf in cases for xlo in (-7.0, -9.0) for kind in "UF" for k in kgrid]
    with Pool(12) as pool:
        R = pd.DataFrame(pool.map(root, rjobs)); S = pd.DataFrame(pool.map(scan, sjobs))
    pd.set_option("display.width", 250)
    ff = lambda v: f"{v:.5f}"
    s = "ROOTS (kind U = issuance kappa*_E, F = burn share kappa*_omega)\n" + R.to_string(index=False, float_format=ff)
    piv = R.pivot_table(index=["psi", "rf", "form", "x_lo"], columns="kind", values="kstar")
    piv["F/U"] = piv["F"] / piv["U"]; piv["F>U"] = piv["F"] > piv["U"]
    s += "\n\n" + piv.to_string(float_format=ff)
    s += "\n\nSCAN of G(kappa) under constant centre (D0: lower-end Gordon denominator of the kappa baseline)\n"
    s += S.pivot_table(index=["psi", "rf", "kind", "kappa"], columns="x_lo", values="G").to_string(float_format=ff)
    if "err" in S: s += "\n\nfailures:\n" + S[S.err.notna()][["psi", "rf", "x_lo", "kind", "kappa", "err"]].to_string(index=False)
    s += "\n\nD0 by kappa:\n" + S[S.kind == "U"].pivot_table(index=["psi", "rf", "kappa"], columns="x_lo", values="D0").to_string(float_format=ff)
    print(s); open("output/check_v5_chtform.txt", "w").write(__doc__ + "\n" + s + "\n")
