"""All sensitivities for the E2' kill test (writes output/dose_sensitivity.txt). Not decisive; the verdict is
dose_killtest.py at capture 0.554."""
import pandas as pd
from dose_killtest import pooled

base = pd.read_csv("e2_windows.csv")[["event", "window", "token", "dpi"]]
W0 = base.merge(pd.read_csv("dose_firm.csv"), on=["event", "window"])

def show(lab, W):
    print(f"{lab:78s} " + " | ".join(f"{pooled(W, c)[0]:6.2f}" for c in (0.554, 0.62, 0.89)))

def mod(W, mask, **kv):
    W = W.copy()
    for k, v in kv.items():
        W.loc[mask(W), k] = v
    return W

print(f"{'reading':78s}  c=0.554 | c=0.62 | c=0.89")
show("firmed (verdict)", W0)
show("#313 read as carrying out the 3-month trial", mod(W0, lambda w: w.window == "#313 vote end", multiplier=0.054))
show("Osmosis at the doc-implied split (0.87%)", mod(W0, lambda w: w.event == "osmo-903", dose=-0.0087))
show("#313 vote-end news share 0.3", mod(W0, lambda w: w.window == "#313 vote end", dpi=0.3))
show("#304 vote end read as permanent (DRC 10-30, Nethermind 11-05 public, no term)",
     mod(W0, lambda w: w.window == "#304 vote end", multiplier=1.0))
show("forum window at 0.054 x #304 dose instead of 0",
     mod(W0, lambda w: w.window == "forum proposal", dose=-0.02672))
for p in (0.5, 0.65, 0.8):
    m = 0.054 + 0.946 * p
    a = mod(W0, lambda w: w.window == "#304 vote end", multiplier=m)
    b = mod(a, lambda w: w.window == "forum proposal", dose=-0.02672, multiplier=m)
    c = mod(b, lambda w: w.event == "osmo-903", dose=-0.0087)
    show(f"renewal prior p={p}: #304 window", a)
    show(f"renewal prior p={p}: + forum window cut at #304 dose", b)
    show(f"renewal prior p={p}: + Osmosis 0.87%", c)
show("upper bound: every dYdX window permanent, forum at #304 dose (contradicts documents)",
     mod(mod(W0, lambda w: w.event == "dydx-313", multiplier=1.0), lambda w: w.window == "forum proposal", dose=-0.02672))
