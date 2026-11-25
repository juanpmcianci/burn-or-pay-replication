"""Gate E2 power calculation, as pre-registered in E2_PLAN.md.

Reads e2_windows.csv (one row per news window: event, token, window_date, first_window_date, dose, dpi) and
computes, for each window, sigma of non-overlapping 3-day abnormal log returns (token minus b*BTC) over the 180 days
ending 30 days before the event's first window. It never touches prices after that cut-off, so no event-window
return is computed.

MDE per window = 2.80 * sigma / (|dose| * dpi); pooled MDE = 2.80 / sqrt(sum((|dose|*dpi/sigma)^2)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRICES = Path(__file__).resolve().parents[2] / "policy-price-impact" / "data" / "prices"
Z = 2.80


def load(token: str) -> pd.Series:
    if token == "CCI30":
        d = pd.read_parquet(PRICES / "cci30.parquet")
    else:
        d = pd.read_parquet(PRICES / f"binance_{token}USDT.parquet")
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    return d.set_index("date")["close"].astype(float).sort_index()


def sigma_abn(token: str, first_window: pd.Timestamp, win: int = 3, lookback: int = 180, gap: int = 30,
              market: str = "BTC") -> tuple[float, float, int]:
    end = first_window - pd.Timedelta(days=gap)
    start = end - pd.Timedelta(days=lookback)
    p, m = load(token), load(market)
    df = pd.concat([np.log(p), np.log(m)], axis=1, keys=["p", "m"], sort=True).loc[start:end].dropna()
    if df.index.max() > end:
        raise AssertionError("look-ahead")
    agg = df.iloc[::win]                      # non-overlapping win-day steps
    r = agg.diff().dropna()
    b = np.cov(r["p"], r["m"])[0, 1] / np.var(r["m"], ddof=1)
    abn = r["p"] - b * r["m"]
    return float(abn.std(ddof=1)), float(b), len(abn)


def run(windows: pd.DataFrame, win=3, market="BTC", dpi_mult=1.0, dose_mult=1.0, dose_col="dose_high") -> tuple[pd.DataFrame, float]:
    rows = []
    for _, w in windows.iterrows():
        s, b, n = sigma_abn(w.token, pd.Timestamp(w.first_window_date), win=win, market=market)
        dpi = min(1.0, w.dpi * dpi_mult)
        dose = abs(w[dose_col]) * dose_mult
        rows.append(dict(event=w.event, window=w.window, token=w.token, sigma=s, beta=b, n=n, dose=w[dose_col] * dose_mult,
                         dpi=dpi, mde=Z * s / (dose * dpi), info=(dose * dpi / s) ** 2))
    out = pd.DataFrame(rows)
    pooled = Z / np.sqrt(out["info"].sum())
    return out, pooled


def verdict(mde: float) -> str:
    return "PASS" if mde <= 0.5 else ("BOUNDS" if mde <= 4.4 else "FAIL")


if __name__ == "__main__":
    wpath = Path(__file__).with_name("e2_windows.csv")
    W = pd.read_csv(wpath)
    pd.set_option("display.width", 200)
    for col in ["dose_high", "dose_low"]:
        base, pooled = run(W, dose_col=col)
        print(f"Central case, {col} (3-day windows, BTC market model):")
        print(base.drop(columns="info").round(5).to_string(index=False))
        print(f"Pooled MDE = {pooled:.2f} years -> {verdict(pooled)}\n")
        ex = base.groupby("event")["info"].sum()
        print("Pooled MDE by event:", {e: round(Z / np.sqrt(v), 2) for e, v in ex.items()}, "\n")
    print("Sensitivity at dose_high (not used for the verdict):")
    for label, kw in [("dpi halved", dict(dpi_mult=0.5)), ("dpi doubled", dict(dpi_mult=2.0)),
                      ("dose -50%", dict(dose_mult=0.5)), ("dose +50%", dict(dose_mult=1.5)),
                      ("5-day windows", dict(win=5)), ("CCi30 market", dict(market="CCI30"))]:
        _, pm = run(W, **kw)
        print(f"  {label:14s} pooled MDE = {pm:7.2f} years -> {verdict(pm)}")
    sys.exit(0)
