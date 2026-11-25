import numpy as np, pandas as pd
o = pd.read_parquet("output/uq1c_draws.parquet")
q = lambda s: np.round(s.dropna().quantile([.05, .25, .5, .75, .95]).values, 2)
print("status:", o.status.value_counts().to_dict())
ok = o[o.status == "ok"].copy()
print("negative slope F:", int((ok.SF <= 0).sum()), "U:", int((ok.SU <= 0).sum()))
print("affinity F (40 subset):", np.round(ok.affF.dropna().describe()[["count","min","max"]].values, 4), "U:", np.round(ok.affU.dropna().describe()[["min","max"]].values, 4))
ok["LR"] = ok.kF / ok.r; ok["ER"] = ok.kU / ok.r; ok["Lss"] = ok.kF / ok.kstar; ok["Ess"] = ok.kU / ok.kstar; ok["LE"] = ok.kF / ok.kU
for c in ["LR", "ER", "Lss", "Ess", "LE"]: print(c, q(ok[c]))
print("share kL > kE:", np.mean(ok.kF > ok.kU).round(3), "| kL > kstar:", np.mean(ok.kF > ok.kstar).round(3), "| kE > kstar:", np.mean(ok.kU > ok.kstar).round(3))
b = ok.r + ok.Gamma / (1 - ok.Theta)
print("kL > r+Gamma/(1-Theta):", np.mean(ok.kF > b).round(3), "min margin", round(float((ok.kF - b).min()), 4), "| kE > bound:", np.mean(ok.kU > b).round(3), "min", round(float((ok.kU-b).min()),4), "| kL>r", np.mean(ok.kF>ok.r))
print("frozen kstar/r on same draws:", q(ok.kstar / ok.r))
for nm, m in [("psi<2", ok.psi < 2), ("psi>=2", ok.psi >= 2)]:
    print(nm, m.sum(), "LR", q(ok.LR[m]), "Lss", q(ok.Lss[m]))
from scipy.stats import spearmanr
print("spearman kL vs kstar", round(spearmanr(ok.kF, ok.kstar)[0], 3), "| spearman Lss vs psi", round(spearmanr(ok.Lss, ok.psi)[0], 3), "vs Theta", round(spearmanr(ok.Lss, ok.Theta)[0],3), "vs Eshare", round(spearmanr(ok.Lss, ok.Eshare)[0],3))
np_ = o[o.status == "no price"]; print("no-price draws: median psi", round(np_.psi.median(),2), "median M", round(np_.M.median(),1), "vs ok M", round(ok.M.median(),1))
