import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
import SimpleITK as sitk

COIF1_LO = np.array([
    -0.0727326195128539,
     0.3378976624578092,
     0.8525720202122554,
     0.38486484686420286,
    -0.0727326195128539,
    -0.01565572813546454,
], dtype=np.float32)

COIF1_HI = np.array([
     0.01565572813546454,
    -0.0727326195128539,
    -0.38486484686420286,
     0.8525720202122554,
    -0.3378976624578092,
    -0.0727326195128539,
], dtype=np.float32)

MC_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)

MC_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

GLCM_OFFSETS = (
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
)


def volume_ml(mask, spacing_xyz):
    vox = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    return float(np.count_nonzero(mask)) * vox / 1000.0


def mean_sd_hu(ct, mask):
    vals = ct[mask.astype(bool)].astype(np.float64)
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0


def dice_and_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    na, nb = int(a.sum()), int(b.sum())
    inter = int(np.logical_and(a, b).sum())
    if na + nb == 0:
        return 1.0, 1.0
    return 2.0 * inter / (na + nb), inter / (na + nb - inter)


def boundary_mask(m):
    er = ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return m & ~er


def surface_distances(a, b, spacing_xyz):
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return np.nan, np.nan
    coords = np.argwhere(a | b)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = coords.max(axis=0) + 3
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    ac, bc = np.pad(a[sl], 1), np.pad(b[sl], 1)
    sp_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    sa, sb = boundary_mask(ac), boundary_mask(bc)
    dt_b = ndimage.distance_transform_edt(~sb, sampling=sp_zyx)
    dt_a = ndimage.distance_transform_edt(~sa, sampling=sp_zyx)
    d_ab, d_ba = dt_b[sa], dt_a[sb]
    hd95 = max(float(np.percentile(d_ab, 95)), float(np.percentile(d_ba, 95)))
    assd = float((d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size))
    return hd95, assd


def mc_area_table(sx, sy, sz):
    scale = np.array([sx, sy, sz], dtype=float)
    v_phys = MC_VERTS * scale
    lut = np.zeros(256, dtype=float)
    for k in range(1, 255):
        bits = np.array([(k >> i) & 1 for i in range(8)], dtype=int)
        pts = [0.5 * (v_phys[u] + v_phys[v]) for u, v in MC_EDGES if bits[u] != bits[v]]
        if len(pts) < 3:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        grad = np.sum((bits[:, None] - 0.5) * (MC_VERTS - 0.5), axis=0) / scale
        n_norm = np.linalg.norm(grad)
        if n_norm < 1e-12:
            _, _, vh = np.linalg.svd(pts_arr - pts_arr.mean(axis=0))
            n_vec = vh[0]
        else:
            n_vec = grad / n_norm
        c = pts_arr.mean(axis=0)
        rel = pts_arr - c
        u_ax = rel[0] - np.dot(rel[0], n_vec) * n_vec
        u_norm = np.linalg.norm(u_ax)
        if u_norm < 1e-12:
            continue
        u_ax /= u_norm
        v_ax = np.cross(n_vec, u_ax)
        angles = np.arctan2(rel @ v_ax, rel @ u_ax)
        pts_ord = pts_arr[np.argsort(angles)]
        area = 0.0
        for i in range(len(pts_ord)):
            p1 = pts_ord[i] - c
            p2 = pts_ord[(i + 1) % len(pts_ord)] - c
            area += 0.5 * float(np.linalg.norm(np.cross(p1, p2)))
        lut[k] = area
    return lut


def sphericity_feature(mask_crop, spacing_xyz, area_lut):
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    n_vox = int(np.count_nonzero(mask_crop))
    if n_vox == 0:
        return np.nan
    m = np.pad(mask_crop.astype(np.uint8), 1)
    idx = (
        m[:-1, :-1, :-1]
        | (m[:-1, :-1, 1:] << 1)
        | (m[:-1, 1:, 1:] << 2)
        | (m[:-1, 1:, :-1] << 3)
        | (m[1:, :-1, :-1] << 4)
        | (m[1:, :-1, 1:] << 5)
        | (m[1:, 1:, 1:] << 6)
        | (m[1:, 1:, :-1] << 7)
    )
    surface_area = float(np.bincount(idx.ravel(), minlength=256) @ area_lut)
    if surface_area <= 0:
        return np.nan
    vol_mm3 = float(n_vox) * sx * sy * sz
    return float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_area)


def discretize_fixed_bin(arr, mask, bin_width=25.0):
    vals = arr[mask]
    mn = float(vals.min())
    low_bound = mn - (mn % bin_width)
    out = np.zeros(arr.shape, dtype=np.int32)
    out[mask] = np.floor((vals - low_bound) / bin_width).astype(np.int32) + 1
    return out


def glszm_glnun_feature(w_hll, mask, bin_width=25.0):
    binned = discretize_fixed_bin(w_hll, mask, bin_width)
    struct26 = np.ones((3, 3, 3), dtype=np.int8)
    u_vals = np.unique(binned[mask])
    if u_vals.size == 0:
        return np.nan
    counts = [float(ndimage.label(binned == int(g), structure=struct26)[1]) for g in u_vals]
    zc = np.asarray(counts, dtype=np.float64)
    tot = float(zc.sum())
    return float(np.sum(zc ** 2) / (tot ** 2)) if tot > 0 else np.nan


def glcm_cluster_prominence_feature(w_llh, mask, bin_width=25.0):
    binned = discretize_fixed_bin(w_llh, mask, bin_width)
    nz, ny, nx = binned.shape
    vals = []
    for dz, dy, dx in GLCM_OFFSETS:
        z0, y0, x0 = slice(max(0, -dz), nz - max(0, dz)), slice(max(0, -dy), ny - max(0, dy)), slice(max(0, -dx), nx - max(0, dx))
        z1, y1, x1 = slice(max(0, dz), nz - max(0, -dz)), slice(max(0, dy), ny - max(0, -dy)), slice(max(0, dx), nx - max(0, -dx))
        valid = mask[z0, y0, x0] & mask[z1, y1, x1]
        if not valid.any():
            continue
        s = binned[z0, y0, x0][valid] + binned[z1, y1, x1][valid]
        cnt = np.bincount(s)
        idx = np.nonzero(cnt)[0].astype(np.float64)
        w = cnt[cnt > 0].astype(np.float64)
        w /= w.sum()
        mu2 = float(np.sum(idx * w))
        vals.append(float(np.sum(((idx - mu2) ** 4) * w)))
    return float(np.mean(vals)) if vals else np.nan


def extract_consensus_features(ct, masks, spacing_xyz):
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        union |= m.astype(bool)
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(axis=0) - 6, 0)
    hi = np.minimum(coords.max(axis=0) + 7, np.array(ct.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

    ct_c = ct[sl].astype(np.float32)
    masks_c = {k: m[sl].astype(bool) for k, m in masks.items()}

    L_y = ndimage.convolve1d(ct_c, COIF1_LO, axis=1, mode="nearest")
    LL_xy = ndimage.convolve1d(L_y, COIF1_LO, axis=2, mode="nearest")
    HL_xy = ndimage.convolve1d(L_y, COIF1_HI, axis=2, mode="nearest")

    w_LLL = ndimage.convolve1d(LL_xy, COIF1_LO, axis=0, mode="nearest")
    w_LLH = ndimage.convolve1d(LL_xy, COIF1_HI, axis=0, mode="nearest")
    w_HLL = ndimage.convolve1d(HL_xy, COIF1_LO, axis=0, mode="nearest")

    area_lut = mc_area_table(spacing_xyz[0], spacing_xyz[1], spacing_xyz[2])
    out = {}
    names = list(masks_c.keys())
    for i, name in enumerate(names):
        mc = masks_c[name]
        reused = False
        for prev in names[:i]:
            if np.array_equal(mc, masks_c[prev]):
                out[name] = dict(out[prev])
                reused = True
                break
        if reused:
            continue
        out[name] = {
            "sphericity": sphericity_feature(mc, spacing_xyz, area_lut),
            "wLLL_p10": float(np.percentile(w_LLL[mc], 10)),
            "wHLL_glnun": glszm_glnun_feature(w_HLL, mc, 25.0),
            "wLLH_cp": glcm_cluster_prominence_feature(w_LLH, mc, 25.0),
        }
    return out


def compute_icc(Y, alpha=0.05):
    Y = np.asarray(Y, dtype=float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    grand = Y.mean()
    ss_tot = ((Y - grand) ** 2).sum()
    ss_r = k * ((Y.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((Y.mean(axis=0) - grand) ** 2).sum()
    ss_e = ss_tot - ss_r - ss_c
    msr = ss_r / (n - 1)
    msc = ss_c / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    q = 1.0 - alpha / 2.0

    df1, df2 = n - 1, (n - 1) * (k - 1)
    icc3 = (msr - mse) / (msr + (k - 1) * mse) if (msr + (k - 1) * mse) != 0 else np.nan
    if mse > 0:
        f3 = msr / mse
        fl3 = f3 / stats.f.ppf(q, df1, df2)
        fu3 = f3 * stats.f.ppf(q, df2, df1)
        lo3, hi3 = (fl3 - 1) / (fl3 + k - 1), (fu3 - 1) / (fu3 + k - 1)
    else:
        lo3, hi3 = 1.0, 1.0

    denom2 = msr + (k - 1) * mse + k * (msc - mse) / n
    icc2 = (msr - mse) / denom2 if denom2 != 0 else np.nan
    if mse > 0 and icc2 < 1.0:
        a = k * icc2 / (n * (1.0 - icc2))
        b = 1.0 + k * icc2 * (n - 1) / (n * (1.0 - icc2))
        v = ((a * msc + b * mse) ** 2) / (((a * msc) ** 2) / (k - 1) + ((b * mse) ** 2) / ((n - 1) * (k - 1)))
        fl2 = stats.f.ppf(q, n - 1, v)
        fu2 = stats.f.ppf(q, v, n - 1)
        d_com = k * msc + (k * n - k - n) * mse
        lo2 = n * (msr - fl2 * mse) / (fl2 * d_com + n * msr)
        hi2 = n * (fu2 * msr - mse) / (d_com + n * fu2 * msr)
    else:
        lo2, hi2 = 1.0, 1.0

    return {
        "ICC2_1": icc2, "ICC2_lo": lo2, "ICC2_hi": hi2,
        "ICC3_1": icc3, "ICC3_lo": lo3, "ICC3_hi": hi3,
    }


def bland_altman_stats(x_second, x_first):
    d = np.asarray(x_second, float) - np.asarray(x_first, float)
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def evaluate_case(case_dir, mask_files):
    ct_img = sitk.ReadImage(str(case_dir / "ct.nii.gz"))
    ct = sitk.GetArrayFromImage(ct_img)
    sp = ct_img.GetSpacing()
    masks = {
        label: sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / fname))) > 0
        for label, fname in mask_files.items()
    }
    rad = extract_consensus_features(ct, masks, sp)
    row = {"case": case_dir.name}
    for label, m in masks.items():
        mu, sd = mean_sd_hu(ct, m)
        row[f"{label}_volume_ml"] = volume_ml(m, sp)
        row[f"{label}_mean_hu"] = mu
        row[f"{label}_sd_hu"] = sd
        for fk, fv in rad[label].items():
            row[f"{label}_{fk}"] = fv

    for a, b in [("TS", "R1"), ("TS", "R2"), ("R1", "R2")]:
        di, io = dice_and_iou(masks[a], masks[b])
        hd95, assd = surface_distances(masks[a], masks[b], sp)
        va, vb = row[f"{a}_volume_ml"], row[f"{b}_volume_ml"]
        row[f"{a}_{b}_dice"] = di
        row[f"{a}_{b}_iou"] = io
        row[f"{a}_{b}_hd95_mm"] = hd95
        row[f"{a}_{b}_assd_mm"] = assd
        row[f"{a}_{b}_vol_diff_pct"] = (vb - va) / (0.5 * (va + vb)) * 100.0
    return row


def main():
    parser = argparse.ArgumentParser(description="Segmentation reproducibility and ICC evaluation.")
    parser.add_argument("--cases_dir", required=True, help="Directory containing per-case NIfTI folders.")
    parser.add_argument("--ts_mask", default="TS.nii.gz", help="Filename for automated TotalSegmentator mask.")
    parser.add_argument("--r1_mask", default="R1.nii.gz", help="Filename for Reader 1 mask.")
    parser.add_argument("--r2_mask", default="R2.nii.gz", help="Filename for Reader 2 mask.")
    parser.add_argument("--outdir", default=".", help="Output directory for CSV summaries.")
    args = parser.parse_args()

    mask_files = {"TS": args.ts_mask, "R1": args.r1_mask, "R2": args.r2_mask}
    root = Path(args.cases_dir)
    case_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "ct.nii.gz").exists()])

    rows = [evaluate_case(d, mask_files) for d in case_dirs]
    df = pd.DataFrame(rows)

    variables = [
        ("volume_ml", "Liver volume (mL)"),
        ("mean_hu", "Mean attenuation (HU)"),
        ("sphericity", "original_shape_Sphericity"),
        ("wLLL_p10", "wavelet-LLL_firstorder_10Percentile"),
        ("wHLL_glnun", "wavelet-HLL_glszm_GrayLevelNonUniformityNormalized"),
        ("wLLH_cp", "wavelet-LLH_glcm_ClusterProminence"),
    ]
    comparisons = [
        ("TS vs R1", ["TS", "R1"]),
        ("TS vs R2", ["TS", "R2"]),
        ("R1 vs R2", ["R1", "R2"]),
        ("TS, R1, R2 (all three)", ["TS", "R1", "R2"]),
    ]

    icc_rows = []
    for key, label in variables:
        for comp_name, rater_list in comparisons:
            cols = [f"{r}_{key}" for r in rater_list]
            res = compute_icc(df[cols].to_numpy(float))
            entry = {"variable": label, "comparison": comp_name, "n": len(df), **res}
            if len(cols) == 2:
                bias, lo, hi = bland_altman_stats(df[cols[1]], df[cols[0]])
                entry.update({"BA_bias": bias, "BA_loa_lo": lo, "BA_loa_hi": hi})
            icc_rows.append(entry)

    icc_df = pd.DataFrame(icc_rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "segmentation_measurements.csv", index=False)
    icc_df.to_csv(out / "segmentation_icc_summary.csv", index=False)
    print(icc_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
