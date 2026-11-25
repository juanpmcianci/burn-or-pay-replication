"""Review round 4, item 24: the event-window abnormal returns of the three governance votes, computed after the
pre-registered power gate (E2) failed, for disclosure. Same design as e2_power.py: 3-day windows starting on the
window date, market model on BTC with beta and sigma from the 180 days ending 30 days before the first window.
Implied gap per window: G_hat = AR / (dose * dpi), with standard error sigma / (|dose| * dpi). Pooled: inverse-variance
weighted, treating windows as independent (they are not for the same event, so the pooled interval is optimistic).
Sign convention: dose is the change in yield paid to stakers (negative = toward burn); G > 0 means payout beats burn,
so a burn move (dose < 0) with G > 0 predicts a negative AR."""
import numpy as np, pandas as pd
from e2_power import load, sigma_abn
W = pd.read_csv("e2_windows.csv")
rows = []
for _, w in W.iterrows():
    s, b, n = sigma_abn(w.token, pd.Timestamp(w.first_window_date))
    p, m = np.log(load(w.token)), np.log(load("BTC"))
    t0 = pd.Timestamp(w.window_date) - pd.Timedelta(days=1)   # close before the window
    t1 = t0 + pd.Timedelta(days=3)
    ar = (p[t1] - p[t0]) - b * (m[t1] - m[t0])
    for dcol in ["dose_high", "dose_low"]:
        k = w[dcol] * w.dpi
        rows.append(dict(event=w.event, window=w.window, dose=dcol, AR=ar, sigma=s, t=ar / s, G=ar / k, se=s / abs(k)))
o = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(o.round(4).to_string(index=False))
for dcol in ["dose_high", "dose_low"]:
    g = o[o.dose == dcol]
    wts = 1 / g.se ** 2
    G = (g.G * wts).sum() / wts.sum(); se = 1 / np.sqrt(wts.sum())
    print(f"pooled {dcol}: G = {G:.1f} years, se {se:.1f}, 95% [{G - 1.96 * se:.1f}, {G + 1.96 * se:.1f}]")
    for e, ge in g.groupby("event"):
        we = 1 / ge.se ** 2; Ge = (ge.G * we).sum() / we.sum(); see = 1 / np.sqrt(we.sum())
        print(f"   {e}: G = {Ge:.1f}, se {see:.1f}, 95% [{Ge - 1.96 * see:.1f}, {Ge + 1.96 * see:.1f}]")
o.to_csv("output/e2_returns.csv", index=False)
