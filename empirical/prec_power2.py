"""Precision check, revision 1: rule-corrected doses (prec_corrections.csv), by-kind split and every sensitivity
named in PREC_PLAN amendment C4, from one script (writes output/prec_power2.txt). Uses output/prec_inputs_merged.csv
(pre-event sd_h and doses only; no CAR fields)."""
import numpy as np
import pandas as pd

Z = 2.80
d0 = pd.read_csv("output/prec_inputs_merged.csv")
corr = pd.read_csv("prec_corrections.csv")
d = d0.merge(corr, on="num", how="left")
d["V"] = np.where(d.V_corrected.notna(), d.V_corrected, d.V_primary)
d["kind2"] = np.where(d.kind_corrected.notna(), d.kind_corrected, d.kind)
P = d[~d.borderline].copy()

def mde(x, V="V", mult=None):
    x = x.dropna(subset=[V])
    v = x[V].abs() * (mult if mult is not None else 1.0)
    info = (v * x.dpi / (x.sd_h * np.sqrt(24))) ** 2
    return Z / np.sqrt(info.sum()), info

m0, _ = mde(P, "V_primary"); m1, info = mde(P)
print(f"primary 24h: as coded {m0:.3f} -> rule-corrected {m1:.3f}")
w = info; neff = w.sum() ** 2 / (w ** 2).sum()
print(f"Kish N_eff {neff:.1f} of {len(w)}; share by token:",
      (w.groupby(P.loc[w.index, 'symbol']).sum() / w.sum()).sort_values(ascending=False).head(4).round(3).to_dict())
grp = np.where(P.kind2 == "emission_cut", "emission", "holder flow")
nz = P.V.fillna(0) != 0
for g in ("holder flow", "emission"):
    s = P[(grp == g) & nz]
    print(f"  {g:12s} n={len(s):2d}  MDE {mde(s)[0]:.3f}")
def alt(num, val):
    x = P.copy(); x.loc[x.num == num, "V"] = val; return mde(x)[0]
print("C4 sensitivities (on corrected doses):")
for lab, num, val in (("Sky #26 realised rate (borderline row)", None, None),
                      ("GMX #32 gross (0.62)", 32, 0.62), ("GMX #33 gross (0.92)", 33, 0.92),
                      ("JUP #16 one-year life (0.33)", 16, 0.33), ("Pump #48 permanent", 48, -0.166 * 4.44 / 0.888),
                      ("Frax #42 at DefiLlama revenue (0.02)", 42, 0.02), ("STRK #59 at realistic share (-0.21 x 0.25/0.654)", 59, -0.08),
                      ("GMX #34 as a pause re-dated (-0.596)", 34, -0.596), ("GMX #35 at current-regime residual (0.26)", 35, 0.26)):
    if num is None:
        print(f"  {lab:50s} n/a (not in the primary sample)"); continue
    print(f"  {lab:50s} {alt(num, val):.3f}")
for k in (0.5, 1.5):
    print(f"  doses x{k:<4} {mde(P, mult=k)[0]:.3f}")
order = info.sort_values(ascending=False)
toks = (info.groupby(P.loc[info.index, 'symbol']).sum()).sort_values(ascending=False)
print(f"  drop top token ({toks.index[0]}): {mde(P[P.symbol != toks.index[0]])[0]:.3f};",
      f"drop top 2 tokens: {mde(P[~P.symbol.isin(toks.index[:2])])[0]:.3f};",
      f"drop top 4 events: {mde(P.drop(order.index[:4]))[0]:.3f}")
