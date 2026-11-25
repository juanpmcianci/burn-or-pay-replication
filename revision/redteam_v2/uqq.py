"""RT v2: UQ draws with the reference at a quarter of F_max (as in Table 5) instead of the midpoint."""
import sys, numpy as np, pandas as pd
R=__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..","..")
sys.path.insert(0,R+"/revision"); sys.path.insert(0,R+"/gonogo"); sys.path.insert(0,R+"/uq")
from uq1 import params
from centred import gap, with_
from multiprocessing import Pool
def one(a):
    i,row=a
    p=params(row,0.0); F=p.F_max/2
    p=with_(p,F_max=4*F,x0=-0.75*np.log(3))
    out=dict(i=i)
    try:
        for k in "FU":
            g0=gap(p,0.0,k,kappa=0.0); s=(gap(p,0.0,k,kappa=0.2)-g0)/0.2
            out["q"+k]=-g0/s
        out["qstatus"]="ok"
    except Exception as e:
        out["qstatus"]="no price" if "no price" in str(e) else "err"
    return out
if __name__=="__main__":
    o=pd.read_parquet(R+"/revision/output/uq1c_draws.parquet"); ok=o[o.status=="ok"].sample(64,random_state=5)
    with Pool(4) as pool: res=pool.map(one,list(ok.iterrows()))
    d=ok.join(pd.DataFrame(res).set_index("i"))
    print(d.qstatus.value_counts().to_dict())
    g=d[d.qstatus=="ok"]
    q=lambda s: np.round(s.quantile([.05,.25,.5,.75,.95]).values,2)
    print("n",len(g)); print("mid L/E",q(g.kF/g.kU)," quarter L/E",q(g.qF/g.qU))
    print("mid L/ss",q(g.kF/g.kstar)," quarter L/ss",q(g.qF/g.kstar))
    print("mid E/ss",q(g.kU/g.kstar)," quarter E/ss",q(g.qU/g.kstar))
    print("all-64 mid L/E",q(d.kF/d.kU))
    print("psi of no-price under quarter:",d[d.qstatus!="ok"].psi.round(2).tolist())
