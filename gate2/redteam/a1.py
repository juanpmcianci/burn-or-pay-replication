import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, pandas as pd
from hetero import *
df=pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','output','sweep2.csv'))
print("surplus_s>0:",(df.G_s_surplus>0).sum(),"max",df.G_s_surplus.max().round(2))
print("surplus_h>0:",(df.G_h_surplus>0).sum())
# dollar reward pool rho*M and staked value Theta*M, per unit dE, normalized
rho0 = df.F + 0.03  # E=0.03, w_B=0
df['dlog_rewardpool']=1/rho0 + df.G_M
df['dlog_stakedvalue']=df.dTheta_drho/df.Theta + df.G_M
print("reward pool $ rises under payout:",(df.dlog_rewardpool>0).sum(),"of",len(df))
print("staked $ value rises under payout:",(df.dlog_stakedvalue>0).sum())
print(df[['dlog_rewardpool','dlog_stakedvalue']].describe().round(2))
print("corner lam=.45 dTheta/drho:",df[df.lam==0.45].dTheta_drho.round(2).agg(['min','max']).tolist())
# entrant-staker criterion: surplus only per capita
print(df[df.G_s_surplus>0][['r_f','psi','F','lam','kappa','G_M','G_s_surplus']].round(2).to_string())
