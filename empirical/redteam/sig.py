import pandas as pd, numpy as np
from pathlib import Path
H=Path(__file__).parent/"h"
PEERS={"OSMO":["ATOM","INJ","TIA","SEI","KAVA"],"DYDX":["GMX","SNX","INJ","JUP","PERP"],"SOL":["AVAX","NEAR","APT","SUI"]}
def ld(t,s): return np.log(pd.read_parquet(H/f"{t}_{s}.parquet")["close"])
for tok,peers in PEERS.items():
    P=pd.concat({s:ld(tok,s) for s in [tok,"BTC","ETH"]+peers},axis=1).dropna()
    print(f"\n{tok}  hours={len(P)}")
    for k in [1,4,12,24,72]:
        r=P.iloc[::k].diff().dropna()
        y=r[tok]; m=r["BTC"]
        b=np.cov(y,m)[0,1]/m.var(); a1=y-b*m
        X=np.column_stack([np.ones(len(r)),r["BTC"],r["ETH"],r[peers].mean(axis=1)])
        n=len(r)//2
        coef=np.linalg.lstsq(X[:n],y.values[:n],rcond=None)[0]   # fit first half, evaluate second half
        a2=y.values[n:]-X[n:]@coef
        mad=lambda v:1.4826*np.median(np.abs(v-np.median(v)))
        print(f" k={k:3d}h n={len(r):5d} sd_btc={a1.std():.4f} mad_btc={mad(a1.values):.4f} sd_peer_oos={a2.std():.4f} kurt={pd.Series(a1).kurt():.1f}")
