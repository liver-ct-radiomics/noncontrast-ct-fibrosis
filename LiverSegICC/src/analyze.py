"""Analysis: compares the TS, AUTH1 and AUTH2 masks.

Usage:  python analyze.py [--workers N]
Only cases that BOTH readers marked as done are included (at least 3 are needed for ICC).

Output (workspace/results/<timestamp>/):
  measurements.csv      per case: liver volume (mL), mean and SD density (HU) of TS, AUTH1, AUTH2
  overlap_per_case.csv  per case: Dice, IoU, HD95, ASSD, volume difference for each pair
  icc_summary.csv       volume and mean density: ICC(2,1), ICC(3,1) with 95% CI, Bland-Altman,
                        for TS vs AUTH1, TS vs AUTH2, AUTH1 vs AUTH2 (and all three masks together)
  overlap_summary.csv   Dice, IoU, HD95, ASSD, volume difference summaries for each pair
  case_details.csv      reader effort (active time, voxels added/removed, strokes) and CT protocol
  report.html           readable report with a draft methods/results paragraph
  bland_altman_*.png    Bland-Altman plots
  excel/                the same tables for Excel in comma-decimal locales (; separator, decimal comma)
"""

import argparse
import html
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import sitk_io
from cases import case_dir, list_cases, read_json, reader_status, write_json
from icc_stats import bland_altman, icc_all, koo_li_category
from metrics import consensus_radiomics_for_case, density, dice_iou, summarize, surface_distances, volume_ml

ALL_THREE = "TS, AUTH1, AUTH2 (all three)"
VOLUME = "Liver volume (mL)"
DENSITY = "Mean density (HU)"
SPHERICITY = "original_shape_Sphericity"
WLLL_P10 = "wavelet-LLL_firstorder_10Percentile"
WHLL_GLNUN = "wavelet-HLL_glszm_GrayLevelNonUniformityNormalized"
WLLH_CP = "wavelet-LLH_glcm_ClusterProminence"

ICC_VARS = (
    ("volume_ml", VOLUME),
    ("mean_hu", DENSITY),
    ("sphericity", SPHERICITY),
    ("wLLL_p10", WLLL_P10),
    ("wHLL_glnun", WHLL_GLNUN),
    ("wLLH_cp", WLLH_CP),
)


def pair_label(a, b):
    return f"{a} vs {b}"


def prog(pct, msg):
    print(f"PROG|{int(pct)}|{msg}", flush=True)


# case selection
def complete_cases():
    ok, skipped = [], []
    for cid in list_cases():
        d = case_dir(cid)
        has = all((d / C.mask_file(m)).exists() for m in C.MASKS)
        done = all(reader_status(cid, r).get("status") == C.DONE for r in C.READERS)
        (ok if has and done else skipped).append(cid)
    return ok, skipped


# per-case metrics
def case_metrics(cid):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    d = case_dir(cid)
    ct_img = sitk_io.read_image(d / C.CT_FILE)
    ct = sitk.GetArrayFromImage(ct_img)
    sp = ct_img.GetSpacing()
    masks = {m: sitk.GetArrayFromImage(sitk_io.read_image(d / C.mask_file(m))) > 0 for m in C.MASKS}
    meta = read_json(d / C.META_FILE, {})
    row = {"case": cid}
    rad = consensus_radiomics_for_case(ct, masks, sp)
    for m, a in masks.items():
        mu, sd = density(ct, a)
        row[f"{m}_volume_ml"] = volume_ml(a, sp)
        row[f"{m}_mean_hu"] = mu
        row[f"{m}_sd_hu"] = sd
        for fk, fv in rad[m].items():
            row[f"{m}_{fk}"] = fv
    for a, b in C.PAIRS:
        di, io = dice_iou(masks[a], masks[b])
        sd_ = surface_distances(masks[a], masks[b], sp)
        k = f"{a}_{b}"
        row[f"{k}_dice"] = di
        row[f"{k}_iou"] = io
        row[f"{k}_hd95_mm"] = sd_["hd95"]
        row[f"{k}_assd_mm"] = sd_["assd"]
        va, vb = row[f"{a}_volume_ml"], row[f"{b}_volume_ml"]
        row[f"{k}_volume_diff_pct"] = (vb - va) / ((va + vb) / 2) * 100 if va + vb else float("nan")
    vox_ml = sp[0] * sp[1] * sp[2] / 1000.0
    for r in C.READERS:
        log = read_json(d / C.reader_log_file(r), {})
        row[f"{r}_active_min"] = float(log.get("active_seconds", float("nan"))) / 60.0
        row[f"{r}_added_ml"] = float(log.get("voxels_added_vs_ts", float("nan"))) * vox_ml
        row[f"{r}_removed_ml"] = float(log.get("voxels_removed_vs_ts", float("nan"))) * vox_ml
        row[f"{r}_strokes"] = log.get("strokes")
    for key in ("series_desc", "slice_thickness_mm", "kvp", "kernel", "manufacturer", "model"):
        row[f"meta_{key}"] = meta.get(key)
    row["meta_spacing_mm"] = "x".join(f"{v:.3f}" for v in (meta.get("spacing_xyz") or list(sp)))
    return row


# summaries
def overlap_summary(df):
    rows = []
    for a, b in C.PAIRS:
        for met, lab in (("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)"), ("assd_mm", "ASSD (mm)"),
                         ("volume_diff_pct", "Volume difference (%)")):
            rows.append({"comparison": pair_label(a, b), "metric": lab, **summarize(df[f"{a}_{b}_{met}"])})
    return pd.DataFrame(rows)


def icc_summary(df):
    rows = []
    for var, lab in ICC_VARS:
        comps = [(pair_label(a, b), [f"{a}_{var}", f"{b}_{var}"]) for a, b in C.PAIRS]
        comps.append((ALL_THREE, [f"{m}_{var}" for m in C.MASKS]))
        for name, cc in comps:
            r = icc_all(df[cc].to_numpy(float))
            row = {
                "variable": lab, "comparison": name, "n": r["n"],
                "ICC2_1": r["ICC2"]["icc"], "ICC2_1_lo": r["ICC2"]["lo"], "ICC2_1_hi": r["ICC2"]["hi"],
                "ICC3_1": r["ICC3"]["icc"], "ICC3_1_lo": r["ICC3"]["lo"], "ICC3_1_hi": r["ICC3"]["hi"],
                "p": r["ICC2"]["p"], "category": koo_li_category(r["ICC2"]["icc"]),
            }
            if len(cc) == 2:
                ba = bland_altman(df[cc[1]], df[cc[0]])   # second minus first
                row.update({"BA_bias": ba["bias"], "BA_loa_lo": ba["loa_lo"], "BA_loa_hi": ba["loa_hi"],
                            "BA_bias_pct": ba["bias_pct"]})
            rows.append(row)
    return pd.DataFrame(rows)


# output
def write_tables(out, tables):
    (out / "excel").mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        df.to_csv(out / "excel" / f"{name}.csv", index=False, sep=";", decimal=",", float_format="%.4f",
                  encoding="utf-8-sig")


def plots(out, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for var, title, unit, name in (("volume_ml", "volume", "mL", "bland_altman_volume.png"),
                                   ("mean_hu", "mean density", "HU", "bland_altman_density.png")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, (a, b) in zip(axes, C.PAIRS):
            x1, x2 = df[f"{a}_{var}"].to_numpy(float), df[f"{b}_{var}"].to_numpy(float)
            ba = bland_altman(x2, x1)
            ax.scatter((x1 + x2) / 2, x2 - x1, s=18, color="#1f5aa6")
            for y, ls in ((ba["bias"], "-"), (ba["loa_lo"], "--"), (ba["loa_hi"], "--")):
                if math.isfinite(y):
                    ax.axhline(y, color="#c0392b" if ls == "-" else "#777", ls=ls, lw=1)
            ax.set_title(f"Bland-Altman {title}: {b} - {a}")
            ax.set_xlabel(f"Mean ({unit})")
            ax.set_ylabel(f"Difference ({unit})")
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        files.append(name)
    return files


def fmt(v, nd=3):
    try:
        if v is None or not math.isfinite(float(v)):
            return "-"
        return f"{float(v):.{nd}f}"
    except Exception:
        return html.escape(str(v))


def df_html(df, nd=3):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            tds.append(f"<td>{fmt(v, nd) if isinstance(v, (float, np.floating)) else html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def mean_sd(values, nd=3):
    s = summarize(values)
    return f"{fmt(s['mean'], nd)} ± {fmt(s['sd'], nd)}"


def ci_text(a, b, c):
    return f"{fmt(a)} [{fmt(b)} to {fmt(c)}]"


def write_report(out, df, ov, ic, figs, info):
    n = len(df)
    ts_ver = info.get("ts_version", "?")
    desc = pd.DataFrame([{
        "mask": m,
        "volume mL (mean ± SD)": mean_sd(df[f"{m}_volume_ml"], 1),
        "mean density HU (mean ± SD)": mean_sd(df[f"{m}_mean_hu"], 1),
        "Sphericity (mean ± SD)": mean_sd(df[f"{m}_sphericity"], 4),
        "wLLL 10Percentile (mean ± SD)": mean_sd(df[f"{m}_wLLL_p10"], 2),
        "wHLL GLNUN (mean ± SD)": mean_sd(df[f"{m}_wHLL_glnun"], 4),
        "wLLH ClusterProminence (mean ± SD)": mean_sd(df[f"{m}_wLLH_cp"], 2),
    } for m in C.MASKS])
    ov_show = pd.DataFrame({
        "comparison": ov["comparison"], "metric": ov["metric"], "n": ov["n"].astype(str),
        "mean ± SD": [f"{fmt(a)} ± {fmt(b)}" for a, b in zip(ov["mean"], ov["sd"])],
        "median [IQR]": [f"{fmt(a)} [{fmt(b)} to {fmt(c)}]" for a, b, c in zip(ov["median"], ov["q1"], ov["q3"])],
        "min": ov["min"],
    })
    ic_show = pd.DataFrame({
        "variable": ic["variable"], "comparison": ic["comparison"], "n": ic["n"].astype(str),
        "ICC(2,1) [95% CI]": [ci_text(a, b, c) for a, b, c in zip(ic.ICC2_1, ic.ICC2_1_lo, ic.ICC2_1_hi)],
        "ICC(3,1) [95% CI]": [ci_text(a, b, c) for a, b, c in zip(ic.ICC3_1, ic.ICC3_1_lo, ic.ICC3_1_hi)],
        "category": ic["category"],
        "Bland-Altman bias [LoA]": ["" if not math.isfinite(a) else f"{fmt(a, 2)} [{fmt(b, 2)}; {fmt(c, 2)}]"
                                    for a, b, c in zip(ic.BA_bias, ic.BA_loa_lo, ic.BA_loa_hi)],
    })
    effort = pd.DataFrame([{
        "reader": r,
        "active time, min (mean ± SD)": mean_sd(df[f"{r}_active_min"]),
        "added mL (mean ± SD)": mean_sd(df[f"{r}_added_ml"]),
        "removed mL (mean ± SD)": mean_sd(df[f"{r}_removed_ml"]),
    } for r in C.READERS])
    meta_df = df[[c for c in df.columns if c.startswith("meta_")]].map(lambda v: "?" if pd.isna(v) else str(v))
    proto = meta_df.agg(" | ".join, axis=1).value_counts()
    proto_html = "".join(f"<li>{html.escape(k)}: {v} case(s)</li>" for k, v in proto.items())

    def dice_txt(a, b, met="dice"):
        s = summarize(df[f"{a}_{b}_{met}"])
        return f"{s['mean']:.3f} ± {s['sd']:.3f}"

    def icc_txt(var, comp):
        r = ic[(ic.variable == var) & (ic.comparison == comp)].iloc[0]
        return f"{r.ICC2_1:.3f} (95% CI {r.ICC2_1_lo:.3f} to {r.ICC2_1_hi:.3f})"

    pair = {(a, b): pair_label(a, b) for a, b in C.PAIRS}
    icc_sentences = "; ".join(
        f"{pair[p]}: volume {icc_txt(VOLUME, pair[p])}, mean density {icc_txt(DENSITY, pair[p])}" for p in C.PAIRS)
    draft = (
        f"<p><b>Methods.</b> Automated whole-liver masks were generated in {n} CT examinations with TotalSegmentator "
        f"(v{html.escape(str(ts_ver))}, 'total' task, liver class). Two readers independently corrected each mask, "
        "blinded to each other's corrections, using a slice-wise brush and eraser editor. Liver volume and mean "
        "density were measured on the automated mask and on each reader's mask. Spatial agreement was quantified "
        "with the Dice similarity coefficient and intersection over union (IoU). Agreement of volume and mean density "
        "was assessed with the intraclass correlation coefficient (two-way random effects, absolute agreement, single "
        "measurement; ICC(2,1)) and interpreted according to Koo and Li.</p>"
        "<p><b>Results.</b> Dice (mean ± SD) was "
        f"{dice_txt('TS', 'AUTH1')} for TS vs AUTH1, {dice_txt('TS', 'AUTH2')} for TS vs AUTH2 and "
        f"{dice_txt('AUTH1', 'AUTH2')} for AUTH1 vs AUTH2; IoU was {dice_txt('TS', 'AUTH1', 'iou')}, "
        f"{dice_txt('TS', 'AUTH2', 'iou')} and {dice_txt('AUTH1', 'AUTH2', 'iou')}, respectively. "
        f"ICC(2,1) was, for {icc_sentences}.</p>"
    )
    css = """body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1200px;color:#222}
    h1{font-size:22px} h2{font-size:18px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}
    table{border-collapse:collapse;margin:8px 0;font-size:13px} th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
    th{background:#f2f4f7} .warn{color:#b03a2e} img{max-width:100%;border:1px solid #ddd;margin:6px 0}
    .draft{background:#f7f9fc;border:1px solid #dde3ea;padding:10px 14px}"""
    figs_html = "".join(f"<img src='{f}' alt='{f}'>" for f in figs)
    skipped = info.get("skipped", [])
    skipped_html = ("<p class='warn'>Not included (not yet done by both readers): "
                    + html.escape(", ".join(skipped)) + "</p>") if skipped else ""
    mixed_html = ("<p class='warn'>More than one TotalSegmentator version was used across these cases.</p>"
                  if "," in str(ts_ver) else "")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Liver segmentation reproducibility</title>
<style>{css}</style></head><body>
<h1>Liver segmentation reproducibility: TS, AUTH1, AUTH2</h1>
<p>Created: {info['created']} · Cases: <b>{n}</b> · TotalSegmentator {html.escape(str(ts_ver))} ·
Device: {html.escape(str(info.get('ts_devices', '?')))}</p>
{skipped_html}{mixed_html}
<h2>1. Liver volume and mean density</h2>{df_html(desc)}
<p>Per-case values are in <code>measurements.csv</code>.</p>
<h2>2. ICC: volume and mean density</h2>
<p>Primary: ICC(2,1), two-way random effects, absolute agreement, single measurement.
Bland-Altman bias = second mask minus first.</p>{df_html(ic_show)}
{figs_html}
<h2>3. Overlap: Dice and IoU (with boundary distances)</h2>
<p>Dice and IoU are pairwise overlap measures, reported as a distribution over cases.
Per-case values are in <code>overlap_per_case.csv</code>.</p>{df_html(ov_show)}
<h2>4. Reader correction effort</h2>{df_html(effort)}
<h2>5. CT protocol</h2><ul>{proto_html}</ul>
<h2>6. Draft text</h2><div class="draft">{draft}</div>
<p style="color:#777;font-size:12px">ICC: Shrout &amp; Fleiss 1979; McGraw &amp; Wong 1996; interpretation Koo &amp; Li 2016
(J Chiropr Med 15:155). HD95: larger of the two directed 95th percentiles of surface distance.</p>
</body></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 4)))
    args = ap.parse_args(argv)

    C.ensure_dirs()
    cids, skipped = complete_cases()
    if len(cids) < 3:
        print(f"FAIL|ICC needs at least 3 cases done by both readers (now {len(cids)})", flush=True)
        return 1
    out = C.RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    prog(2, f"Analysing {len(cids)} cases")

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(cids)))) as ex:
        futs = {ex.submit(case_metrics, c): c for c in cids}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            prog(2 + 80 * i / len(cids), f"Measured {i}/{len(cids)} ({futs[f]})")
    df = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    ov = overlap_summary(df)
    ic = icc_summary(df)

    prog(90, "Writing tables and report")
    measure_cols = [
        f"{m}_{v}" for m in C.MASKS
        for v in ("volume_ml", "mean_hu", "sd_hu", "sphericity", "wLLL_p10", "wHLL_glnun", "wLLH_cp")
    ]
    pair_cols = [f"{a}_{b}_{v}" for a, b in C.PAIRS
                 for v in ("dice", "iou", "hd95_mm", "assd_mm", "volume_diff_pct")]
    tables = {
        "measurements": df[["case", *measure_cols]],
        "overlap_per_case": df[["case", *pair_cols]],
        "icc_summary": ic,
        "overlap_summary": ov,
        "case_details": df[["case", *[c for c in df.columns if c not in measure_cols + pair_cols + ["case"]]]],
    }
    write_tables(out, tables)
    figs = plots(out, df)
    ts_infos = [read_json(case_dir(c) / C.TS_INFO_FILE, {}) or {} for c in cids]
    versions = {i.get("totalsegmentator_version") for i in ts_infos}
    devices = pd.Series([i.get("device") or "?" for i in ts_infos]).value_counts()
    info = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ts_version": ", ".join(sorted(v for v in versions if v)) or "?",
        "ts_devices": ", ".join(f"{k} ({v})" for k, v in devices.items()),
        "skipped": skipped,
    }
    write_report(out, df, ov, ic, figs, info)
    write_json(out / "run_info.json", {**info, "cases": cids})

    for a, b in C.PAIRS:
        lab = pair_label(a, b)
        iv = ic[(ic.variable == VOLUME) & (ic.comparison == lab)].iloc[0]
        id_ = ic[(ic.variable == DENSITY) & (ic.comparison == lab)].iloc[0]
        print(f"RESULT|{lab}:  ICC volume {iv.ICC2_1:.3f} [{iv.ICC2_1_lo:.3f} to {iv.ICC2_1_hi:.3f}]   "
              f"ICC density {id_.ICC2_1:.3f} [{id_.ICC2_1_lo:.3f} to {id_.ICC2_1_hi:.3f}]   "
              f"Dice {mean_sd(df[f'{a}_{b}_dice'])}   IoU {mean_sd(df[f'{a}_{b}_iou'])}", flush=True)
    print(f"DONE|{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
