"""Gate 2 level 0: the pass test's +-50% box, plus the corner where stakers are fully staked."""
import numpy as np, pandas as pd
from hetero import P2, calibrate, gaps
pd.set_option("display.width", 250)
rows = []
for r_f in [0.06, 0.10]:
  for psi in [1.1, 1.5, 2.0, 3.0]:
    for F in [0.015, 0.03, 0.045]:
      for lam in [0.45, 0.5, 0.6, 0.75, 0.9, 1.0]:
        for k in [0.0, 0.1]:
          try:
            p = calibrate(P2(psi=psi, F=F, kappa=k, r_f=r_f), staker_wealth=lam)
            g = gaps(p)
          except Exception as e:
            continue
          g.update(r_f=r_f, psi=psi, F=F, lam=lam, kappa=k); rows.append(g)
df = pd.DataFrame(rows)
df["conflict"] = (df.G_s > 0) & (df.G_M < 0)
df["tot_vs_M"] = np.sign(df.G_tot) != np.sign(df.G_M)
df.to_csv("output/sweep2.csv", index=False)
print(len(df), "feasible cases;", int(df.conflict.sum()), "with stakers preferring payout while market cap prefers burn;",
      int(df.tot_vs_M.sum()), "with total welfare and market cap disagreeing;", int((df.G_M > 0).sum()), "with G_M > 0;",
      int((df.G_h > 0).sum()), "with holders preferring payout")
cols = ["r_f","psi","F","lam","kappa","theta_s","Gamma_s","G_M","G_s","G_s_endow","G_s_surplus","G_h","G_tot"]
print(df[df.conflict | df.tot_vs_M][cols].round(2).to_string(index=False))
print("\nG_s by psi and lam (kappa=0, F=0.03, r_f=0.06 / 0.10):")
print(df[(df.kappa==0)&(df.F==0.03)].pivot_table(index=["r_f","psi"], columns="lam", values="G_s").round(1).to_string())
print("\nG_M by psi and lam (kappa=0, F=0.03):")
print(df[(df.kappa==0)&(df.F==0.03)].pivot_table(index=["r_f","psi"], columns="lam", values="G_M").round(1).to_string())
