"""Gate E2' step 0: capture share of tokenomics news in the first 4 hours (pre-registered in E2P_PLAN.md).

Input: e2p_events.csv with columns event, symbol (Binance, e.g. UNIUSDT), t0_utc (ISO, minute precision).
For each event: hourly Binance closes for token, BTC and ETH over [t0 - 31d, t0 + 2d]; market model on
BTC and ETH estimated over [t0 - 30d, t0 - 2d]; CAR over hours 0..3, 0..23 and -24..23; noise from residuals.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # run with policy-price-impact/.venv/bin/python (system Python lacks requests and its CA chain)

HERE = Path(__file__).parent
CACHE = HERE / "output" / "hourly_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def klines(sym: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Series:
    f = CACHE / f"{sym}_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.parquet"
    if f.exists():
        return pd.read_parquet(f)["close"]
    t, end, rows = int(t0.value // 10**6), int(t1.value // 10**6), []
    while t < end:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params=dict(symbol=sym, interval="1h", startTime=t, endTime=end, limit=1000), timeout=30).json()
        if not isinstance(r, list) or not r:
            break
        rows += r
        t = r[-1][0] + 3_600_000
        if len(r) < 1000:
            break
        time.sleep(0.05)
    s = pd.Series([float(x[4]) for x in rows], index=pd.to_datetime([x[0] for x in rows], unit="ms"), name="close")
    s.to_frame().to_parquet(f)
    return s


def one(event: str, sym: str, t0: pd.Timestamp) -> dict:
    h0 = t0.floor("h")
    a, b = h0 - pd.Timedelta(days=31), h0 + pd.Timedelta(days=2)
    px = pd.concat({k: klines(s, a, b) for k, s in (("tok", sym), ("btc", "BTCUSDT"), ("eth", "ETHUSDT"))},
                   axis=1, sort=True).dropna()
    # Binance klines are indexed by OPEN time and `close` is the price at the top of the next hour, so
    # log(close).diff() at index i is the return from the top of hour i to the top of hour i+1: hour i itself.
    # (Revision 1: draft 0 shifted this back one hour, which moved the announcement hour into the pre-window.)
    r = np.log(px).diff().dropna()
    est = r.loc[h0 - pd.Timedelta(days=30): h0 - pd.Timedelta(days=2)]
    X = np.column_stack([np.ones(len(est)), est["btc"], est["eth"]])
    coef, *_ = np.linalg.lstsq(X, est["tok"].values, rcond=None)
    sd = float(np.std(est["tok"].values - X @ coef, ddof=3))
    Xa = np.column_stack([np.ones(len(r)), r["btc"], r["eth"]])
    ab = pd.Series(r["tok"].values - Xa @ coef, index=r.index)
    k = ((ab.index - h0) / pd.Timedelta(hours=1)).astype(int)
    ab = pd.Series(ab.values, index=k)

    def car(lo, hi):
        w = ab.loc[lo:hi]
        if len(w) != hi - lo + 1:
            return np.nan
        return float(w.sum())
    return dict(event=event, symbol=sym, t0=t0, n_est=len(est), sd_h=sd, b_btc=coef[1], b_eth=coef[2],
                car4=car(0, 3), car24=car(0, 23), pre24=car(-24, -1), total=car(-24, 23),
                sig4=sd * 2, sig48=sd * np.sqrt(48))


def summarise(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> dict:
    d = df.dropna(subset=["car4", "total"]).copy()
    d["qual"] = d.total.abs() >= 2 * d.sig48
    d["c"] = d.car4 / d.total
    q = d[d.qual]
    med = float(q.c.median()) if len(q) else np.nan

    def cstar(x):
        return ((x.car4 * x.total - x.sig4 ** 2).sum()) / ((x.total ** 2 - x.sig48 ** 2).sum())
    rng = np.random.default_rng(seed)
    boots = [cstar(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    bq = [q.c.iloc[rng.integers(0, len(q), len(q))].median() for _ in range(n_boot)] if len(q) else [np.nan]
    return dict(n_events=len(d), n_qual=len(q), median_c=med, median_c_ci=tuple(np.nanpercentile(bq, [5, 95])),
                c_star=float(cstar(d)), c_star_ci=tuple(np.percentile(boots, [5, 95])), table=d)


def verdict(s: dict) -> str:
    if s["n_qual"] < 8:
        return "INCONCLUSIVE"
    c = s["median_c"]
    return "GO" if c >= 0.85 else ("CONDITIONAL" if c >= 0.54 else "STOP")


if __name__ == "__main__":
    ev = pd.read_csv(HERE / "e2p_events.csv")
    rows = [one(r.event, r.symbol, pd.Timestamp(r.t0_utc).tz_localize(None) if pd.Timestamp(r.t0_utc).tzinfo is None
                else pd.Timestamp(r.t0_utc).tz_convert("UTC").tz_localize(None)) for r in ev.itertuples()]
    s = summarise(pd.DataFrame(rows))
    pd.set_option("display.width", 220)
    t = s["table"]
    print(t[["event", "symbol", "t0", "sd_h", "pre24", "car4", "car24", "total", "sig48", "qual", "c"]].round(4).to_string(index=False))
    print(f"\nevents {s['n_events']}, qualifying {s['n_qual']}")
    print(f"median capture (qualifying) = {s['median_c']:.3f}, bootstrap 90% {np.round(s['median_c_ci'], 3)}")
    print(f"noise-corrected pooled c* (all events) = {s['c_star']:.3f}, bootstrap 90% {np.round(s['c_star_ci'], 3)}")
    print("VERDICT:", verdict(s))
