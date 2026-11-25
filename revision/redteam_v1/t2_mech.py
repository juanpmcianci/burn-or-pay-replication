"""T2: why does feedback help payout less under the burn share?
Slopes dG/dkappa (k 0 -> k1) for burn share (F) and issuance (U) under:
  full      : model as in the paper
  basefix   : base path has no feedback (kb=0); feedback acts only on the policy-induced stake
  driftpin  : feedback acts on growth (mu_A w) but Theta pinned in the state-motion term (mu_A w_z)
and for fee schedules of different speed s (fees fall faster for small s)."""
from rt import *

def th0(p):
    return lambda z: theta_of_rho(rho_of(z, p), p)

def slopes(psi, rf, k1, **cal):
    p0 = make(psi, rf, 0.0, **cal)
    x = x_at_theta(p0)
    p1 = with_(p0, kappa=k1)
    r = {}
    g0F, g0U = gap(p0, x, "F"), gap(p0, x, "U")
    r["G0"] = (g0F, g0U)
    for lab, kw in [("full", {}), ("basefix", dict(kb=0.0, th_base=th0(p0))), ("driftpin", dict(drift_pin=th0(p0)))]:
        gF, gU = gap(p1, x, "F", **kw), gap(p1, x, "U", **kw)
        r[lab] = ((gF - g0F) / k1, (gU - g0U) / k1)
    return x, fee(np.array([x]), p0)[0], r

if __name__ == "__main__":
    cases = [(1.1, 0.10, 0.1, {}), (1.5, 0.20, 0.2, {}), (1.1, 0.30, 0.2, {})]
    cases += [(1.5, 0.20, 0.2, dict(s=s)) for s in [0.4, 1.5, 3.0]]
    cases += [(1.5, 0.20, 0.2, dict(F_max=0.06)), (1.5, 0.20, 0.2, dict(F_max=0.24))]
    for psi, rf, k1, cal in cases:
        try:
            x, F, r = slopes(psi, rf, k1, **cal)
            s = " | ".join(f"{k}: F={v[0]:7.2f} U={v[1]:7.2f} F/U={v[0]/v[1]:.3f}" for k, v in r.items())
            print(f"psi={psi} rf={rf} k1={k1} {cal} x={x:.2f} F={F:.3f}: {s}", flush=True)
        except Exception as e:
            print(psi, rf, cal, "err", str(e)[:60], flush=True)
