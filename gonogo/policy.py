"""Proposition 5, part (ii): policy derivatives in the CHT economy.

Pay-versus-burn. Moving a constant revenue yield b from payout to burn sets
rho -> rho - b and E_net -> E_net - b, which is exactly an emission cut of b
(Prop. 2, token payout = burn + issuance). So the gap of payout over burn, per
unit of revenue yield, is

    G2(x) = d log W(x) / dE        [years; same units as V1's G and A(k)]

The direct cash-flow effects (Gamma += b versus -E_net += b) cancel exactly, so
G2 is the stake effect alone. The model's own value of stake, per unit Theta, is

    V_true(x) = G2(x) / (dTheta(x)/drho).

Check: an outside subsidy D gives dlogW/dD = direct + stake, and with Theta pinned
in every channel it gives the direct part only; their difference must equal G2.

Channels: pin all but one channel at the baseline Theta path to split G2 into
Gamma (lock-up dissipation), active network, and productivity feedback.
"""
from __future__ import annotations

import numpy as np

from cht import calibrated, solve, theta_of_rho, rho_of, with_

H = 2e-4          # finite-difference step in yield units
XS = np.linspace(-3, 3, 13)


def dlogW(p, field, pin=None, xs=XS, h=H):
    up = solve(with_(p, **{field: getattr(p, field) + h}), pin=pin)
    dn = solve(with_(p, **{field: getattr(p, field) - h}), pin=pin)
    return (up.logW(xs) - dn.logW(xs)) / (2 * h)


def dtheta_drho(p, xs=XS, h=H):
    r = rho_of(xs, p)
    return (theta_of_rho(r + h, p) - theta_of_rho(r - h, p)) / (2 * h)


def baseline_pin(p):
    """Callables returning the baseline Theta path, for pinning channels."""
    f = lambda x: theta_of_rho(rho_of(x, p), p)
    return f


def run(p, xs=XS):
    th0 = baseline_pin(p)
    out = {"x": xs, "Theta": th0(xs), "dTheta_drho": dtheta_drho(p, xs)}
    out["G2"] = dlogW(p, "E", xs=xs)
    all_pinned = {"gamma": th0, "net": th0, "fb": th0}
    out["dD_total"] = dlogW(p, "D", xs=xs)
    out["dD_direct"] = dlogW(p, "D", pin=all_pinned, xs=xs)
    for ch in ["gamma", "net", "fb"]:
        pin = {k: th0 for k in ["gamma", "net", "fb"] if k != ch}
        out[f"G2_{ch}"] = dlogW(p, "E", pin=pin, xs=xs)
    out["V_true"] = out["G2"] / out["dTheta_drho"]
    for ch in ["gamma", "net", "fb"]:
        out[f"V_{ch}"] = out[f"G2_{ch}"] / out["dTheta_drho"]
    return out


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    for beta in [1.1, 1.05]:
        for k in [0.0, 0.05, 0.1]:
            p = calibrated(beta=beta, kappa=k, F_max=0.12, F_min=0.0)
            o = run(p)
            df = pd.DataFrame(o)
            df["check"] = df.dD_total - df.dD_direct - df.G2
            print(f"\n=== beta={beta} kappa={k} ===")
            print(df[["x", "Theta", "dTheta_drho", "G2", "G2_gamma", "G2_net", "G2_fb",
                      "V_true", "V_gamma", "V_net", "V_fb", "dD_direct", "check"]].round(3).to_string(index=False))
