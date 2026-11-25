"""Kill test for E2' (DOSE_PLAN.md): pooled 4-hour, peer-controlled MDE with firmed doses.

Noise: 4-hour abnormal-return sd, peer basket fitted out of sample, pre-event samples only (redteam/sig.py,
rerun in the session: OSMO 0.0096, DYDX 0.0107, SOL 0.0158 vs BTC only).
MDE = 2.80 / sqrt(sum((|dose| * dpi * capture / sigma)^2)).
Firmed doses are read from dose_firm.csv: event, window, dose (annual, signed), multiplier (duration).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SIG4H = {"OSMO": 0.0096, "DYDX": 0.0107, "SOL": 0.0158}
Z = 2.80


def pooled(W: pd.DataFrame, capture: float) -> tuple[float, pd.DataFrame]:
    W = W.copy()
    W["eff_dose"] = W["dose"].abs() * W["multiplier"]
    W["info"] = (W["eff_dose"] * W["dpi"] * capture / W["token"].map(SIG4H)) ** 2
    W["mde_window"] = Z / np.sqrt(W["info"])
    return Z / np.sqrt(W["info"].sum()), W


if __name__ == "__main__":
    base = pd.read_csv(HERE / "e2_windows.csv")[["event", "window", "token", "dpi"]]
    firm = pd.read_csv(HERE / "dose_firm.csv")
    W = base.merge(firm, on=["event", "window"], how="left", validate="1:1")
    assert W["dose"].notna().all(), "every window needs a firmed dose"
    pd.set_option("display.width", 200)
    for c in [0.554, 0.62, 0.89]:
        m, t = pooled(W, c)
        tag = "  <- decisive" if c == 0.554 else ""
        print(f"capture {c}: pooled 4h MDE = {m:.2f} years -> {'PROCEED' if m <= 3.9 else 'STOP'}{tag}")
        if c == 0.554:
            print(t[["event", "window", "token", "dose", "multiplier", "eff_dose", "dpi", "mde_window"]].round(5).to_string(index=False))
            print("by event:", {e: round(Z / np.sqrt(v), 2) for e, v in t.groupby("event")["info"].sum().items()})
