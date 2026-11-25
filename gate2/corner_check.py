"""Strict corner: stakers fully staked (theta_s = 1) with slack, so the finite difference does not straddle the kink.
u_s is raised above the calibrated kink value and m_s re-solved so that Theta = 0.45 (hence lambda = 0.45)."""
import numpy as np, pandas as pd
from scipy.optimize import brentq
from hetero import P2, calibrate, gaps, equilibrium, with_
rows = []
for r_f in [0.06, 0.10]:
  for psi in [1.1, 1.5, 2.0, 3.0]:
    for F in [0.015, 0.03, 0.045]:
      for k in [0.0, 0.1]:
        try:
            base = calibrate(P2(psi=psi, F=F, kappa=k, r_f=r_f), staker_wealth=0.45)
        except Exception:
            continue
        for mult in [1.05, 1.5]:
            q0 = with_(base, u_s=base.u_s * mult)
            try:
                ms = brentq(lambda m: equilibrium(with_(q0, m_s=m))["Theta"] - 0.45, 0.05, 0.95, xtol=1e-12)
                q = with_(q0, m_s=ms)
                g = gaps(q)
            except Exception:
                continue
            g.update(r_f=r_f, psi=psi, F=F, kappa=k, mult=mult); rows.append(g)
df = pd.DataFrame(rows)
print(len(df), "strict-corner cases; theta_s min", df.theta_s.min().round(6),
      "| max G_s", df.G_s.max().round(2), "| max G_s surplus", df.G_s_surplus.max().round(2),
      "| max G_M", df.G_M.max().round(2), "| max G_h", df.G_h.max().round(2),
      "| dTheta/drho", df.dTheta_drho.min().round(2), "-", df.dTheta_drho.max().round(2))
print(df.loc[df.G_s.idxmax(), ["r_f","psi","F","kappa","mult","G_M","G_s","G_s_endow","G_s_surplus","G_h"]].round(2).to_dict())
