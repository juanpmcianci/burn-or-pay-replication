"""Robustness: sign of the model's value of stake across lock-cost exponents and feedback strengths.

For each (beta, kappa) report, at the state where Theta = 0.45:
  dTheta/drho (compare CHT Table 2 col. 8: 0.413), G2 = dlogW/dE (pay minus burn, years),
  V_true = G2 / (dTheta/drho), and its channels.
Then kappa* where G2 = 0 (payout ties burn), by bisection on kappa.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from cht import calibrated, solve, theta_of_rho, rho_of, with_
from policy import dlogW, dtheta_drho


def x_at_theta(p, target=0.45):
    return brentq(lambda x: theta_of_rho(rho_of(np.array([x]), p), p)[0] - target, -6, 6)


def at_mid(p, field="E", pin=None):
    x = np.array([x_at_theta(p)])
    return dlogW(p, field, pin=pin, xs=x)[0], dtheta_drho(p, xs=x)[0], x[0]


def channels_mid(p):
    x = np.array([x_at_theta(p)])
    th0 = lambda z: theta_of_rho(rho_of(z, p), p)
    out = {}
    for ch in ["gamma", "net", "fb"]:
        pin = {k: th0 for k in ["gamma", "net", "fb"] if k != ch}
        out[ch] = dlogW(p, "E", pin=pin, xs=x)[0]
    out["direct"] = dlogW(p, "D", pin={k: th0 for k in ["gamma", "net", "fb"]}, xs=x)[0]
    return out


if __name__ == "__main__":
    print("beta  r_f  kappa  dTh/drho   G2(yrs)  V_true   V_gamma  V_net  V_fb   direct(yrs)  r=rho/Theta")
    rows = []
    for beta in [1.1, 1.5, 2.0]:
        for r_f in [0.06, 0.10]:
            for kappa in [0.0, 0.1, 0.2, 0.3]:
                p = calibrated(beta=beta, kappa=kappa, F_max=0.12, F_min=0.0, r_f=r_f)
                try:
                    g2, dth, x = at_mid(p)
                    ch = channels_mid(p)
                except Exception as e:
                    print(f"{beta:4} {r_f:4} {kappa:5}  -- {str(e)[:50]}")
                    continue
                r = rho_of(np.array([x]), p)[0] / 0.45
                print(f"{beta:4} {r_f:4} {kappa:5}  {dth:7.3f}  {g2:8.2f}  {g2/dth:7.2f}  {ch['gamma']/dth:7.2f}"
                      f"  {ch['net']/dth:6.2f}  {ch['fb']/dth:5.2f}   {ch['direct']:8.2f}     {r:.3f}", flush=True)
    print("\nkappa* (payout ties burn at Theta = 0.45), r_f = 0.10:")
    for beta in [1.1, 1.5, 2.0]:
        def g(k):
            return at_mid(calibrated(beta=beta, kappa=k, F_max=0.12, F_min=0.0, r_f=0.10))[0]
        try:
            ks = brentq(g, 0.0, 0.6, xtol=1e-3)
            print(f"  beta={beta}: kappa* = {ks:.3f}  (10 pp more stake -> +{ks*10:.1f} pp/yr productivity growth)")
        except Exception as e:
            print(f"  beta={beta}: no root in [0, 0.6]: {str(e)[:60]}")
