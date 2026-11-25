"""T1: near the existence edge. Does the gap blow-up come from the lower boundary value?"""
from rt import *
from feasible import G_feasible, G_uniform

for psi, rf, ks in [(1.5, 0.15, [0.0, 0.3, 0.40, 0.44, 0.452, 0.456]),
                    (1.1, 0.10, [0.0, 0.1, 0.2, 0.21]),
                    (11.2, 0.15, [0.4, 0.6, 0.8, 0.866, 1.0, 1.2])]:
    for k in ks:
        p = make(psi, rf, k)
        x = x_at_theta(p)
        out = []
        for lab, kw in [("std", {}), ("pinBC", dict(pin_bc=True)), ("xlo-5", dict(x_lo=-5.0)),
                        ("xlo-9", dict(x_lo=-9.0)), ("onesided", dict(one_sided=True))]:
            try:
                gF = gap(p, x, "F", **kw)
                gU = gap(p, x, "U", **{k2: v for k2, v in kw.items() if k2 != "one_sided"})
                out.append(f"{lab}: F={gF:9.3f} U={gU:9.3f}")
            except Exception as e:
                out.append(f"{lab}: err {str(e)[:30]}")
        print(f"psi={psi} rf={rf} k={k}: " + " | ".join(out), flush=True)
