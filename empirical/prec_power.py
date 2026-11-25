"""Precision check (PREC_PLAN.md): pooled minimum detectable capitalisation ratio kappa_V from pre-event noise only.
Noise sd_h comes from e2p_capture.one() (the market-model residual sd over 30 to 2 days before t0); the CAR
fields it also returns are not read or printed here. Run with policy-price-impact/.venv/bin/python."""
import numpy as np
import pandas as pd
import e2p_capture as m

Z = 2.80
doses = pd.concat([pd.read_csv("prec_doses_A.csv"), pd.read_csv("prec_doses_B.csv")])
doses["num"] = doses["num"].astype(int)
ev = pd.read_csv("e2p_events_all.csv")
ev["num"] = ev["event"].str[:2].astype(int)
ev.loc[ev.num == 34, "t0_utc"] = "2026-02-12 22:33"          # amendment C2
df = ev.merge(doses[["num", "kind", "V_primary", "V_sens_face", "dpi"]], on="num", how="left")
sd = []
for r in df.itertuples():
    sd.append(m.one(r.event, r.symbol, pd.Timestamp(r.t0_utc))["sd_h"])   # only sd_h is kept
df["sd_h"] = sd
df["V_primary"] = pd.to_numeric(df["V_primary"], errors="coerce")
df["V_sens_face"] = pd.to_numeric(df["V_sens_face"], errors="coerce")
df.drop(columns=[c for c in df.columns if c.startswith("car")], errors="ignore").to_csv("output/prec_inputs_merged.csv", index=False)

def mde(d, vcol="V_primary", design="24h", c=1.0, dpi_mult=1.0, vmult=1.0):
    d = d.dropna(subset=[vcol])
    sig = d.sd_h * (np.sqrt(24) if design == "24h" else 2.0)
    info = ((d[vcol].abs() * vmult) * (d.dpi * dpi_mult).clip(upper=1) * c / sig) ** 2
    return Z / np.sqrt(info.sum()), len(d), info

def verdict(x):
    return "GO" if x <= 0.5 else ("WEAK" if x <= 1 else "NO")

prim = df[~df.borderline]
print(f"primary events {len(prim)}; with a dose {prim.V_primary.notna().sum()}; NA excluded {prim.V_primary.isna().sum()}")
for lab, d in (("primary 50", prim), ("all 66", df)):
    for design, c in (("24h", 1.0), ("4h", 0.554)):
        x, n, info = mde(d, design=design, c=c)
        print(f"{lab:10s} {design} (capture {c}): MDE_kappa = {x:.3f}  n={n}  -> {verdict(x)}")
x, n, info = mde(prim)
top = prim.dropna(subset=["V_primary"]).assign(share=(info / info.sum()).values).sort_values("share", ascending=False)
print("\nshare of pooled information, primary 24h (top 8):")
print(top[["event", "kind", "V_primary", "dpi", "sd_h", "share"]].head(8).round(4).to_string(index=False))
print("\nsensitivities, primary 24h:")
for lab, kw in (("face value / treasury counted", dict(vcol="V_sens_face")), ("doses -50%", dict(vmult=0.5)),
                ("doses +50%", dict(vmult=1.5)), ("news shares halved", dict(dpi_mult=0.5))):
    print(f"  {lab:32s} MDE_kappa = {mde(prim, **kw)[0]:.3f}")
# drop the single most informative event (leave-one-out max)
loo = [mde(prim.drop(i))[0] for i in prim.dropna(subset=['V_primary']).index]
print(f"  leave-one-out range            {min(loo):.3f} to {max(loo):.3f}")
