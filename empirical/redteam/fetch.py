import requests, pandas as pd, time, sys
from pathlib import Path
OUT=Path(__file__).parent/"h"; OUT.mkdir(exist_ok=True)
SP={"OSMO":("2024-07-10","2025-01-06"),"DYDX":("2025-04-01","2025-09-28"),"SOL":("2023-06-22","2023-11-19")}
PEERS={"OSMO":["BTC","ETH","ATOM","INJ","TIA","SEI","KAVA"],"DYDX":["BTC","ETH","GMX","SNX","INJ","JUP","PERP"],"SOL":["BTC","ETH","AVAX","NEAR","APT","SUI"]}
def get(sym,t0,t1):
    t=int(pd.Timestamp(t0).value//1e6); end=int(pd.Timestamp(t1).value//1e6); rows=[]
    while t<end:
        r=requests.get("https://api.binance.com/api/v3/klines",params=dict(symbol=sym,interval="1h",startTime=t,endTime=end,limit=1000),timeout=30).json()
        if not isinstance(r,list) or not r: break
        rows+=r; t=r[-1][0]+3600000
        if len(r)<1000: break
        time.sleep(0.05)
    df=pd.DataFrame(rows)
    return pd.Series(df[4].astype(float).values,index=pd.to_datetime(df[0],unit="ms")) if rows else pd.Series(dtype=float)
for tok,(a,b) in SP.items():
    for s in [tok]+PEERS[tok]:
        x=get(s+"USDT",a,b); assert len(x)==0 or x.index.max()<=pd.Timestamp(b)
        x.to_frame("close").to_parquet(OUT/f"{tok}_{s}.parquet"); print(tok,s,len(x))
