import numpy as np, pandas as pd
W=pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..","e2_windows.csv"))
base={"OSMO":0.07306,"DYDX":0.05891,"SOL":0.06780}
# sigma by design, from sig.py (pre-event only); SOL peer adj does not help, keep BTC-only
S={"3d BTC (plan)":base,
   "3d peer OOS":{"OSMO":.0526,"DYDX":.0326,"SOL":.0697},
   "24h peer OOS":{"OSMO":.0280,"DYDX":.0217,"SOL":.0389},
   "12h peer OOS":{"OSMO":.0178,"DYDX":.0169,"SOL":.0278},
   "4h peer OOS":{"OSMO":.0096,"DYDX":.0107,"SOL":.0158}}
for col in ["dose_high","dose_low"]:
  for cap in [1.0,0.5]:
    out=[]
    for k,s in S.items():
        f=1.0 if k.startswith("3d") else cap   # share of window news landing inside a short window
        info=((W[col].abs()*W.dpi*f)/W.token.map(s))**2
        out.append(f"{k}: {2.8/np.sqrt(info.sum()):.1f}")
    print(col,"short-window capture",cap," | ".join(out))
# events needed (dYdX-sized) for 4.4 under each design
for k,s in S.items():
    d=W[W.event=="dydx-313"]; i=((d.dose_high.abs()*d.dpi)/s["DYDX"])**2
    print(k,"dYdX alone MDE",round(2.8/np.sqrt(i.sum()),1),"events for 4.4:",round((2.8/4.4)**2/i.sum(),1))
