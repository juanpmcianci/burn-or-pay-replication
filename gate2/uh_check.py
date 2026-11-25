"""Level-1 spot check: holders who stake a little (u_h > 0). Calibrated at u_h = 0, then u_h raised; Theta moves off 0.45."""
from hetero import P2, calibrate, gaps, with_
for psi, r_f in [(1.1, 0.06), (2.0, 0.10)]:
    for lam in [0.5, 0.75]:
        base = calibrate(P2(psi=psi, r_f=r_f), staker_wealth=lam)
        for frac in [0.0, 0.05, 0.2, 0.5]:
            try:
                g = gaps(with_(base, u_h=frac * base.u_s))
                print(f"psi={psi} r_f={r_f} lam={lam} u_h/u_s={frac}: Theta={g['Theta']:.3f} G_M={g['G_M']:.1f} G_s={g['G_s']:.1f} G_h={g['G_h']:.1f} G_tot={g['G_tot']:.1f}")
            except Exception as e:
                print(f"psi={psi} lam={lam} u_h/u_s={frac}: -- {str(e)[:60]}")
