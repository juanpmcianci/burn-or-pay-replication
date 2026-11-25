import requests,pandas as pd,numpy as np
def f(sym,a,b):
    t=int(pd.Timestamp(a).value//1e6);e=int(pd.Timestamp(b).value//1e6);rows=[]
    while t<e:
        r=requests.get("https://fapi.binance.com/fapi/v1/fundingRate",params=dict(symbol=sym,startTime=t,endTime=e,limit=1000),timeout=30).json()
        if not isinstance(r,list) or not r:break
        rows+=r;t=r[-1]["fundingTime"]+1
        if len(r)<1000:break
    d=pd.DataFrame(rows);d["t"]=pd.to_datetime(d.fundingTime,unit="ms");return d.set_index("t").fundingRate.astype(float)
for sym,a,b in [("DYDXUSDT","2025-04-01","2025-09-28"),("OSMOUSDT","2024-07-10","2025-01-06"),("SOLUSDT","2023-06-22","2023-11-19")]:
    x=f(sym,a,b)
    if len(x)==0: print(sym,"none");continue
    per_day=x.resample("1D").sum()*365   # annualised funding, daily
    d3=per_day.rolling(3).mean().iloc[::3]
    print(sym,len(x),"ann. funding mean %.3f sd(daily) %.3f sd(3d avg) %.3f sd(3d change) %.3f"%(per_day.mean(),per_day.std(),d3.std(),d3.diff().std()))
