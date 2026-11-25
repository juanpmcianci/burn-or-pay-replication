import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from hetero import *
from scipy.optimize import brentq
def tot_ex(p,h=1e-5):
    def W(q):
        eq=equilibrium(q); Th=eq['Theta']; M=eq['M']; e={j:v['m']*v['w']/M for j,v in eq['types'].items()}
        tot=sum(v['total'] for v in welfare(q,eq,e).values())
        return tot - q.F*M/(q.r_f-mu_A(q,Th)-q.mu_S), M
    M0=equilibrium(p)['M']
    return (W(with_(p,E=p.E+h))[0]-W(with_(p,E=p.E-h))[0])/(2*h)/M0
df=pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','output','sweep2.csv'))
rows=[]
for _,r in df.iterrows():
    p=calibrate(P2(psi=r.psi,F=r.F,kappa=r.kappa,r_f=r.r_f),staker_wealth=r.lam)
    rows.append(tot_ex(p))
df['G_tot_exfee']=rows
print("box: G_tot incl fee-inflow term vs excluding (fee payers' loss netted):")
print(df[['G_M','G_tot','G_tot_exfee']].describe().round(1))
print("cases G_tot_exfee>0:",(df.G_tot_exfee>0).sum(), " of which G_M<0:",((df.G_tot_exfee>0)&(df.G_M<0)).sum())
print(df[df.G_tot_exfee>0][['r_f','psi','F','lam','kappa','G_M','G_tot','G_tot_exfee']].round(1).head(20).to_string())


for k in [0.1,0.15,0.18,0.2,0.25,0.3]:
    p=calibrate(P2(kappa=k),staker_wealth=.75); g=gaps(p)
    print(k, "G_M",round(g['G_M'],1),"G_tot",round(g['G_tot'],1),"G_tot_exfee",round(tot_ex(p),1))
