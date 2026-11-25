"""Red team v3: audit of Proposition 2 from the paper's own outputs (output/prop2_weights.csv, prop2_check.csv).
For each case: whole-line formula (kX_avg) vs finite-difference hurdle; monotonicity of kstar(z) and of F on the
99% support of pi_E; where kstar(z) is non-monotone; the mean location of pi_E and pi_L; covariance check;
local hurdle at the reference state vs the closed form; q-free 'cost/benefit' representation."""
import numpy as np, pandas as pd
W = pd.read_csv("../output/prop2_weights.csv"); C = pd.read_csv("../output/prop2_check.csv")
rows = []
for case, g in W.groupby("case", sort=False):
    z = g.dz.values; piE = g.piE.values; piL = g.piL.values; ks = g.kstar.values; F = g.F.values; q = g.q.values
    tr = np.trapezoid
    EF = tr(piE * F, z); EK = tr(piE * ks, z)
    cov = tr(piE * F * ks, z) - EF * EK
    kE = EK; kL = tr(piL * ks, z)
    cdf = np.cumsum(piE) / piE.sum(); sel = (cdf > 0.005) & (cdf < 0.995)
    dk = np.diff(ks[sel]); zz = z[sel][1:]
    inc = dk > 1e-9
    # largest increasing stretch of kstar (non-monotone region) and the pi_E mass on it
    massinc = tr(np.where(np.r_[False, dk > 1e-9], piE[sel], 0), z[sel]) if inc.any() else 0.0
    i0 = np.argmin(np.abs(z))
    rows.append(dict(case=case, kE_avg=kE, kL_avg=kL, cov_over_EF=cov / EF, check=kL - kE - cov / EF,
                     kstar_z0=ks[i0], kstar_max=ks[sel].max(), z_at_max=z[sel][ks[sel].argmax()],
                     kstar_monotone=not inc.any(), piE_mass_where_kstar_rises=massinc,
                     z_inc_lo=zz[inc].min() if inc.any() else np.nan, z_inc_hi=zz[inc].max() if inc.any() else np.nan,
                     meanz_E=tr(piE * z, z), meanz_L=tr(piL * z, z), qmin_supp=q[sel].min(),
                     ks_m3=np.interp(-3, z, ks), ks_p3=np.interp(3, z, ks)))
R = pd.DataFrame(rows)
C["key"] = C.apply(lambda r: f"{r.case}|{r.psi}|{r.rf}|" + ("{}" if pd.isna(r.s) else "{'s': " + str(r.s) + "}"), axis=1)
R = R.merge(C[["key", "kE_fd", "kL_fd", "kE_formula", "kL_formula", "kcl_ref"]], left_on="case", right_on="key", how="left")
R["dE_wholeline"] = R.kE_avg - R.kE_fd; R["dL_wholeline"] = R.kL_avg - R.kL_fd
R["dE_bc"] = R.kE_formula - R.kE_fd; R["dL_bc"] = R.kL_formula - R.kL_fd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["case", "kE_fd", "kE_avg", "dE_wholeline", "dE_bc", "kL_fd", "kL_avg", "dL_wholeline", "dL_bc", "cov_over_EF", "check"]
print(R[cols].round(4).to_string(index=False))
cols = ["case", "kcl_ref", "kstar_z0", "ks_m3", "ks_p3", "kstar_max", "z_at_max", "kstar_monotone", "piE_mass_where_kstar_rises", "z_inc_lo", "z_inc_hi", "meanz_E", "meanz_L", "qmin_supp"]
print(R[cols].round(3).to_string(index=False))
print("max |whole-line - fd|: E", R.dE_wholeline.abs().max().round(4), "L", R.dL_wholeline.abs().max().round(4),
      "; relative:", (R.dE_wholeline.abs() / R.kE_fd).max().round(3), (R.dL_wholeline.abs() / R.kL_fd).max().round(3))
print("max |with-BC - fd|:", R.dE_bc.abs().max().round(5), R.dL_bc.abs().max().round(5))
R.to_csv("prop2_audit.csv", index=False)
