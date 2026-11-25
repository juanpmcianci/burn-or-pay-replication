"""Robustness run for gate E2' step 0: both samples, leak share and leave-one-out c* (writes output/e2p_capture_all66.txt).
Run with policy-price-impact/.venv/bin/python."""
import numpy as np
import pandas as pd
import e2p_capture as m

def cs(x):
    return ((x.car4 * x.total - x.sig4 ** 2).sum()) / ((x.total ** 2 - x.sig48 ** 2).sum())

for f in ["e2p_events.csv", "e2p_events_all.csv"]:
    ev = pd.read_csv(f)
    rows = [m.one(r.event, r.symbol, pd.Timestamp(r.t0_utc)) for r in ev.itertuples()]
    s = m.summarise(pd.DataFrame(rows)); t = s["table"]; q = t[t.qual]
    print(f, "events", s["n_events"], "qual", s["n_qual"], "median c", round(s["median_c"], 3), np.round(s["median_c_ci"], 3),
          "c*", round(s["c_star"], 3), np.round(s["c_star_ci"], 3), "->", m.verdict(s))
    print("  median leak share pre24/total (qualifying):", round((q.pre24 / q.total).median(), 3))
    loo = [cs(t.drop(i)) for i in t.index]
    print("  c* leave-one-out range", round(min(loo), 3), round(max(loo), 3))
    if f.endswith("all.csv"):
        print(q[["event", "pre24", "car4", "total", "c"]].round(3).to_string(index=False))
