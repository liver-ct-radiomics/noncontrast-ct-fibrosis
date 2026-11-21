"""Independent check of the calculations. Run: python selftest.py

1) ICC: Shrout & Fleiss (1979) Table 2 example -> ICC(1,1)=0.17, ICC(2,1)=0.29, ICC(3,1)=0.71
   (values printed in the paper); 95% CIs compared with pingouin's output.
2) Dice/IoU: two cubes with a known overlap.
3) HD95/ASSD: a cube shifted by one voxel, so the surface distance is known.
4) Volume: known voxel count x spacing.
"""

import sys

import numpy as np

from icc_stats import bland_altman, icc_all
from metrics import consensus_radiomics_for_case, density, dice_iou, surface_distances, volume_ml

FAILS = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {got:.4f} (expected {want:.4f} +/- {tol})")
    if not ok:
        FAILS.append(name)


print("1) ICC - Shrout & Fleiss 1979 example (6 subjects x 4 raters)")
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
r = icc_all(SF)
check("ICC(1,1)", r["ICC1"]["icc"], 0.17, 0.005)
check("ICC(2,1)", r["ICC2"]["icc"], 0.29, 0.005)
check("ICC(3,1)", r["ICC3"]["icc"], 0.71, 0.005)
check("ICC(1,1) lower CI", r["ICC1"]["lo"], -0.13, 0.01)
check("ICC(1,1) upper CI", r["ICC1"]["hi"], 0.72, 0.01)
check("ICC(2,1) lower CI", r["ICC2"]["lo"], 0.02, 0.01)
check("ICC(2,1) upper CI", r["ICC2"]["hi"], 0.76, 0.01)
check("ICC(3,1) lower CI", r["ICC3"]["lo"], 0.34, 0.01)
check("ICC(3,1) upper CI", r["ICC3"]["hi"], 0.95, 0.01)
check("F(2/3)", r["ICC3"]["F"], 11.03, 0.02)

print("   Perfect agreement (identical columns) -> ICC=1")
same = np.repeat(np.arange(10, dtype=float)[:, None], 3, axis=1)
r = icc_all(same)
check("ICC(2,1) identical", r["ICC2"]["icc"], 1.0, 1e-9)
print("   Constant shift (+5) -> ICC(3,1)=1, ICC(2,1)<1")
shift = np.c_[np.arange(10.0), np.arange(10.0) + 5]
r = icc_all(shift)
check("ICC(3,1) shift", r["ICC3"]["icc"], 1.0, 1e-9)
ok = r["ICC2"]["icc"] < 0.9
print(f"  [{'OK' if ok else 'FAIL'}] ICC(2,1) penalises the shift: {r['ICC2']['icc']:.4f}")
if not ok:
    FAILS.append("ICC2 shift")

ba = bland_altman([10, 12, 14], [9, 11, 13])
check("Bland-Altman bias", ba["bias"], 1.0, 1e-9)

print("2) Dice / IoU")
a = np.zeros((20, 20, 20), bool)
b = np.zeros_like(a)
a[0:10, 0:10, 0:10] = True          # 1000
b[5:15, 0:10, 0:10] = True          # 1000, intersection 500
d, j = dice_iou(a, b)
check("Dice", d, 2 * 500 / 2000, 1e-12)
check("IoU", j, 500 / 1500, 1e-12)

print("3) Surface distances")
a = np.zeros((30, 30, 30), bool)
b = np.zeros_like(a)
a[10:20, 10:20, 10:20] = True
b[10:20, 10:20, 11:21] = True       # 1 voxel shift in x, x spacing 0.8 mm
s = surface_distances(a, b, (0.8, 1.0, 1.0))
check("HD (max)", s["hd"], 0.8, 1e-6)
ok = 0 < s["assd"] < 0.8
print(f"  [{'OK' if ok else 'FAIL'}] ASSD between 0 and 0.8: {s['assd']:.4f}")
if not ok:
    FAILS.append("ASSD")
s0 = surface_distances(a, a, (0.8, 1.0, 1.0))
check("HD95 same mask", s0["hd95"], 0.0, 1e-12)

print("4) Volume and density")
check("Volume mL", volume_ml(a, (0.8, 0.8, 1.25)), 1000 * 0.8 * 0.8 * 1.25 / 1000, 1e-12)
ct = np.full(a.shape, -100, np.int16)
ct[a] = 60
check("Mean HU", density(ct, a)[0], 60.0, 1e-12)

print("5) 4 consensus radiomic features")
rng = np.random.default_rng(42)
ct_rad = rng.normal(50.0, 12.0, size=(30, 30, 30)).astype(np.float32)
rad = consensus_radiomics_for_case(ct_rad, {"TS": a, "AUTH1": a, "AUTH2": b}, (0.8, 0.8, 1.25))
for fk in ("sphericity", "wLLL_p10", "wHLL_glnun", "wLLH_cp"):
    check(f"Identical mask {fk}", rad["TS"][fk], rad["AUTH1"][fk], 1e-12)
ok_sph = 0.75 < rad["TS"]["sphericity"] < 0.90
print(f"  [{'OK' if ok_sph else 'FAIL'}] Cube sphericity in (0.75, 0.90): {rad['TS']['sphericity']:.4f}")
if not ok_sph:
    FAILS.append("sphericity range")

print()
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("All checks passed.")
