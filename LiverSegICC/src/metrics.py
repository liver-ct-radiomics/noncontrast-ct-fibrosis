"""Mask measurements: volume, density, Dice, IoU, HD95, ASSD, and the 4 consensus radiomic features.

Arrays are in SimpleITK order (z, y, x). Spacing is given in mm as (x, y, z) and converted
to (z, y, x) here.
"""

import math
import numpy as np
from scipy import ndimage

# Coiflet-1 (coif1) decomposition low-pass and high-pass filters (PyWavelets / PyRadiomics default)
_COIF1_DEC_LO = np.array([
    -0.0727326195128539,
     0.3378976624578092,
     0.8525720202122554,
     0.38486484686420286,
    -0.0727326195128539,
    -0.01565572813546454,
], dtype=np.float32)

_COIF1_DEC_HI = np.array([
     0.01565572813546454,
    -0.0727326195128539,
    -0.38486484686420286,
     0.8525720202122554,
    -0.3378976624578092,
    -0.0727326195128539,
], dtype=np.float32)

# Unit cube vertices (x, y, z) and 12 edges for Marching-Cubes surface area lookup
_MC_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)

_MC_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# 13 unique 3D symmetric GLCM offsets at Chebyshev distance 1
_GLCM_OFFSETS = (
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
)


def volume_ml(mask, spacing_xyz):
    vox = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    return float(np.count_nonzero(mask)) * vox / 1000.0


def density(ct, mask):
    """Mean and SD of HU inside the mask."""
    vals = ct[mask.astype(bool)]
    if vals.size == 0:
        return float("nan"), float("nan")
    vals = vals.astype(np.float64)
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0


def dice_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    na, nb = int(a.sum()), int(b.sum())
    inter = int(np.logical_and(a, b).sum())
    if na + nb == 0:
        return 1.0, 1.0
    union = na + nb - inter
    return 2.0 * inter / (na + nb), inter / union


def _surface(m):
    # 6-neighbour erosion; outside the volume counts as background, so a mask touching the
    # volume edge still has a surface there
    er = ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return m & ~er


def surface_distances(a, b, spacing_xyz):
    """HD95 (larger of the two directed 95th percentiles, as in MONAI/medpy) and ASSD, in mm."""
    a = a.astype(bool)
    b = b.astype(bool)
    nan = float("nan")
    if not a.any() or not b.any():
        return {"hd95": nan, "hd": nan, "assd": nan}
    union = a | b
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = coords.max(axis=0) + 3
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    a_c = np.pad(a[sl], 1)
    b_c = np.pad(b[sl], 1)
    sp_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    sa, sb = _surface(a_c), _surface(b_c)
    dt_b = ndimage.distance_transform_edt(~sb, sampling=sp_zyx)
    dt_a = ndimage.distance_transform_edt(~sa, sampling=sp_zyx)
    d_ab = dt_b[sa]
    d_ba = dt_a[sb]
    hd95 = max(float(np.percentile(d_ab, 95)), float(np.percentile(d_ba, 95)))
    hd = max(float(d_ab.max()), float(d_ba.max()))
    assd = float((d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size))
    return {"hd95": hd95, "hd": hd, "assd": assd}


def _build_mc_area_lut(sx, sy, sz):
    """256-entry Marching-Cubes surface area lookup table for voxel spacing (sx, sy, sz)."""
    scale = np.array([sx, sy, sz], dtype=float)
    v_phys = _MC_VERTS * scale
    lut = np.zeros(256, dtype=float)
    for k in range(1, 255):
        bits = np.array([(k >> i) & 1 for i in range(8)], dtype=int)
        pts = [0.5 * (v_phys[u] + v_phys[v]) for u, v in _MC_EDGES if bits[u] != bits[v]]
        if len(pts) < 3:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        grad = np.sum((bits[:, None] - 0.5) * (_MC_VERTS - 0.5), axis=0) / scale
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


def _sphericity(mask_crop, spacing_xyz, area_lut):
    """original_shape_Sphericity = (pi^(1/3) * (6 * V)^(2/3)) / A."""
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    n_vox = int(np.count_nonzero(mask_crop))
    if n_vox == 0:
        return float("nan")
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
        return float("nan")
    vol_mm3 = float(n_vox) * sx * sy * sz
    return float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_area)


def _bin_image(arr, mask, bin_width = 25.0):
    """PyRadiomics imageoperations.binImage discretization (binWidth = 25.0)."""
    vals = arr[mask]
    mn = float(vals.min())
    low_bound = mn - (mn % bin_width)
    out = np.zeros(arr.shape, dtype=np.int32)
    out[mask] = np.floor((vals - low_bound) / bin_width).astype(np.int32) + 1
    return out


def _glszm_glnun(w_hll, mask, bin_width = 25.0):
    """wavelet-HLL_glszm_GrayLevelNonUniformityNormalized = sum_g (N_g^2) / (sum_g N_g)^2."""
    binned = _bin_image(w_hll, mask, bin_width)
    struct26 = np.ones((3, 3, 3), dtype=np.int8)
    u_vals = np.unique(binned[mask])
    if u_vals.size == 0:
        return float("nan")
    zone_counts = []
    for g in u_vals:
        _, n_z = ndimage.label(binned == int(g), structure=struct26)
        zone_counts.append(float(n_z))
    zc = np.asarray(zone_counts, dtype=np.float64)
    tot = float(zc.sum())
    return float(np.sum(zc ** 2) / (tot ** 2)) if tot > 0 else float("nan")


def _glcm_cluster_prominence(w_llh, mask, bin_width = 25.0):
    """wavelet-LLH_glcm_ClusterProminence averaged across the 13 3D symmetric directions."""
    binned = _bin_image(w_llh, mask, bin_width)
    nz, ny, nx = binned.shape
    cp_vals = []
    for dz, dy, dx in _GLCM_OFFSETS:
        z0 = slice(max(0, -dz), nz - max(0, dz))
        y0 = slice(max(0, -dy), ny - max(0, dy))
        x0 = slice(max(0, -dx), nx - max(0, dx))
        z1 = slice(max(0, dz), nz - max(0, -dz))
        y1 = slice(max(0, dy), ny - max(0, -dy))
        x1 = slice(max(0, dx), nx - max(0, -dx))
        valid = mask[z0, y0, x0] & mask[z1, y1, x1]
        if not valid.any():
            continue
        s = binned[z0, y0, x0][valid] + binned[z1, y1, x1][valid]
        counts = np.bincount(s)
        idx = np.nonzero(counts)[0].astype(np.float64)
        w = counts[counts > 0].astype(np.float64)
        w /= w.sum()
        mu2 = float(np.sum(idx * w))
        cp_vals.append(float(np.sum(((idx - mu2) ** 4) * w)))
    return float(np.mean(cp_vals)) if cp_vals else float("nan")


def consensus_radiomics_for_case(ct, masks, spacing_xyz):
    """Compute the 4 consensus radiomic features for TS, AUTH1, and AUTH2 on a single cropped CT."""
    nan = float("nan")
    empty_feats = {
        "sphericity": nan,
        "wLLL_p10": nan,
        "wHLL_glnun": nan,
        "wLLH_cp": nan,
    }
    union = np.zeros_like( next(iter(masks.values())), dtype=bool )
    for m in masks.values():
        union |= m.astype(bool)
    if not union.any():
        return {k(empty_feats) for k in masks}

    coords = np.argwhere(union)
    pad = 6
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.array(ct.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

    ct_c = ct[sl].astype(np.float32)
    masks_c = {k: m[sl].astype(bool) for k, m in masks.items()}

    # Separable 3D undecimated Coiflet-1 wavelet subbands:
    # PyRadiomics convention on (z, y, x) array: first letter = x (axis 2), second = y (axis 1), third = z (axis 0)
    L_y = ndimage.convolve1d(ct_c, _COIF1_DEC_LO, axis=1, mode="nearest")
    LL_xy = ndimage.convolve1d(L_y, _COIF1_DEC_LO, axis=2, mode="nearest")
    HL_xy = ndimage.convolve1d(L_y, _COIF1_DEC_HI, axis=2, mode="nearest")

    w_LLL = ndimage.convolve1d(LL_xy, _COIF1_DEC_LO, axis=0, mode="nearest")
    w_LLH = ndimage.convolve1d(LL_xy, _COIF1_DEC_HI, axis=0, mode="nearest")
    w_HLL = ndimage.convolve1d(HL_xy, _COIF1_DEC_LO, axis=0, mode="nearest")

    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    area_lut = _build_mc_area_lut(sx, sy, sz)

    out = {}
    names = list(masks_c.keys())
    for idx_m, name in enumerate(names):
        mc = masks_c[name]
        if not mc.any():
            out[name] = dict(empty_feats)
            continue
        # Reuse exact values if this mask is voxel-identical to an already computed mask (e.g., unmodified TS)
        copied = False
        for prev_name in names[:idx_m]:
            if np.array_equal(mc, masks_c[prev_name]):
                out[name] = dict(out[prev_name])
                copied = True
                break
        if copied:
            continue

        out[name] = {
            "sphericity": _sphericity(mc, spacing_xyz, area_lut),
            "wLLL_p10": float(np.percentile(w_LLL[mc], 10)),
            "wHLL_glnun": _glszm_glnun(w_HLL, mc, 25.0),
            "wLLH_cp": _glcm_cluster_prominence(w_LLH, mc, 25.0),
        }
    return out


def summarize(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    nan = float("nan")
    if v.size == 0:
        return {"n": 0, "mean": nan, "sd": nan, "median": nan, "q1": nan, "q3": nan, "min": nan, "max": nan}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q1": float(np.percentile(v, 25)),
        "q3": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }
