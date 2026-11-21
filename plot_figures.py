"""Generate main evaluation figures (Fig 3, Fig 4, Fig 5) and the supplementary
K = 1-5 calibration and decision-curve figure (S1 Fig) from out-of-fold
probability tables produced by nested_cv_pipeline.py.

Usage:
    python plot_figures.py --indir . --outdir .
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)

FRAC = 0.70
N_BOOT_BAND = 500

C_FIB4 = "#64748b"
C_CORE = "#2563eb"
C_REG1 = "#059669"
C_REG2 = "#d97706"
C_REG3 = "#dc2626"


def loess_on_grid(y_true, y_prob, grid, frac=FRAC):
    out = lowess(y_true, y_prob, frac=frac, it=1, return_sorted=True)
    return np.interp(grid, out[:, 0], out[:, 1])


def loess_curve(y_true, y_prob, frac=FRAC):
    grid = np.linspace(y_prob.min(), y_prob.max(), 100)
    return grid, np.clip(loess_on_grid(y_true, y_prob, grid, frac), 0.0, 1.0)


def bootstrap_loess_band(y_true, y_prob, grid, frac=FRAC, n_boot=N_BOOT_BAND, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    curves = np.full((n_boot, len(grid)), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        curves[b] = loess_on_grid(y_true[idx], y_prob[idx], grid, frac)
    return np.nanpercentile(curves, 2.5, axis=0), np.nanpercentile(curves, 97.5, axis=0)


def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    return np.array([
        (np.sum((y_prob >= t) & (y_true == 1)) / n)
        - (np.sum((y_prob >= t) & (y_true == 0)) / n) * (t / (1.0 - t))
        for t in thresholds
    ])


def save_png_and_tif(fig, outdir, stem, pad_inches=0.06):
    png_path = os.path.join(outdir, f"{stem}.png")
    tif_path = os.path.join(outdir, f"{stem}.tif")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    with Image.open(png_path) as im:
        im.convert("RGB").save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))


def plot_fig3_calibration(y, p_core, p_reg1, p_reg2, p_reg3, outdir):
    g_core, l_core = loess_curve(y, p_core)
    g_reg1, l_reg1 = loess_curve(y, p_reg1)
    g_reg2, l_reg2 = loess_curve(y, p_reg2)
    g_reg3, l_reg3 = loess_curve(y, p_reg3)

    lo_r1, hi_r1 = bootstrap_loess_band(y, p_reg1, g_reg1, seed=42)
    lo_r3, hi_r3 = bootstrap_loess_band(y, p_reg3, g_reg3, seed=43)

    fig = plt.figure(figsize=(7.6, 8.0), dpi=300)
    gs = GridSpec(2, 1, height_ratios=[3.8, 1.15], hspace=0.07, figure=fig)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", lw=1.3,
            label="Perfect Calibration", zorder=2)
    ax.fill_between(g_reg1, np.clip(lo_r1, 0, 1), np.clip(hi_r1, 0, 1),
                    color=C_REG1, alpha=0.12, zorder=1, label="95% Bootstrap Band (Regime 1)")
    ax.fill_between(g_reg3, np.clip(lo_r3, 0, 1), np.clip(hi_r3, 0, 1),
                    color=C_REG3, alpha=0.10, zorder=1, label="95% Bootstrap Band (Regime 3)")

    ax.plot(g_core, l_core, color=C_CORE, lw=2.0, linestyle="-",
            label="Selected Clinical Core (FIB-4 + Sex)", zorder=4)
    ax.plot(g_reg1, l_reg1, color=C_REG1, lw=2.0, linestyle=(0, (5, 2)),
            label="Regime 1: Consensus Hybrid (Fixed $K=4$)", zorder=5)
    ax.plot(g_reg3, l_reg3, color=C_REG3, lw=2.2, linestyle="-",
            label="Regime 3: Within-Fold Selection (In-Fold Pre-filter, $K=4$)", zorder=6)
    ax.plot(g_reg2, l_reg2, color=C_REG2, lw=1.9, linestyle=(0, (1.2, 2.2)),
            marker="o", markersize=3.2, markevery=6,
            label="Regime 2: Within-Fold Selection (Cohort Pre-filter, $K=4$)", zorder=7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Observed Fraction of Significant Fibrosis", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=0)
    ax.legend(bbox_to_anchor=(0.5, 1.015), loc="lower center", ncol=2, fontsize=7.8,
              handlelength=2.2, borderpad=0.5, labelspacing=0.4, columnspacing=1.2,
              frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.98)
    plt.setp(ax.get_xticklabels(), visible=False)

    bins = np.linspace(0.0, 1.0, 31)
    axh.hist(p_core, bins=bins, color=C_CORE, alpha=0.22, zorder=2)
    axh.hist(p_core, bins=bins, histtype="step", color=C_CORE, lw=1.5, zorder=3)
    axh.hist(p_reg1, bins=bins, color=C_REG1, alpha=0.22, zorder=4)
    axh.hist(p_reg1, bins=bins, histtype="step", color=C_REG1, lw=1.5, linestyle=(0, (5, 2)), zorder=5)
    axh.hist(p_reg3, bins=bins, color=C_REG3, alpha=0.18, zorder=6)
    axh.hist(p_reg3, bins=bins, histtype="step", color=C_REG3, lw=1.6, linestyle="-", zorder=7)
    axh.hist(p_reg2, bins=bins, histtype="step", color=C_REG2, lw=1.6, linestyle=(0, (1.2, 2.2)), zorder=8)

    axh.set_xlim(0.0, 1.0)
    axh.set_ylim(0, 32)
    axh.set_xlabel("Mean Out-of-Fold Predicted Probability", fontsize=11, labelpad=8)
    axh.set_ylabel("Patient Count", fontsize=10, labelpad=8)
    axh.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=0)

    save_png_and_tif(fig, outdir, "Fig3")
    plt.close(fig)


def plot_fig4_dca(y, p_core, p_reg1, p_reg2, p_reg3, outdir):
    n = len(y)
    thresholds = np.round(np.arange(0.01, 0.905, 0.01), 2)
    nb_all = (np.sum(y == 1) / n) - (np.sum(y == 0) / n) * (thresholds / (1.0 - thresholds))
    nb_none = np.zeros_like(thresholds)
    nb_core = net_benefit(y, p_core, thresholds)
    nb_r1 = net_benefit(y, p_reg1, thresholds)
    nb_r2 = net_benefit(y, p_reg2, thresholds)
    nb_r3 = net_benefit(y, p_reg3, thresholds)

    dca_df = pd.DataFrame({
        "pt": thresholds,
        "nb_all": nb_all,
        "nb_none": nb_none,
        "nb_core": nb_core,
        "nb_reg1": nb_r1,
        "nb_reg2": nb_r2,
        "nb_reg3": nb_r3,
    })
    dca_df.to_csv(os.path.join(outdir, "dca_thresholds.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=300)
    ax.axvspan(0.35, 0.65, color="#f1f5f9", alpha=0.9, zorder=0,
               label="Threshold Range ($p_t$ = 0.35–0.65)")
    ax.axvline(0.35, color="#94a3b8", linestyle="--", lw=1.0, zorder=1)
    ax.axvline(0.65, color="#94a3b8", linestyle="--", lw=1.0, zorder=1)
    ax.plot(thresholds, nb_none, color="#1e293b", lw=1.3, label="Treat None (Baseline)", zorder=2)
    ax.plot(thresholds, nb_all, linestyle=(0, (3, 3)), color="#64748b", lw=1.5, label="Treat All", zorder=2)
    ax.plot(thresholds, nb_core, color=C_CORE, lw=2.0, linestyle="-",
            label="Selected Clinical Core (FIB-4 + Sex)", zorder=4)
    ax.plot(thresholds, nb_r1, color=C_REG1, lw=2.0, linestyle=(0, (5, 2)),
            label="Regime 1: Consensus Hybrid (Fixed $K=4$)", zorder=5)
    ax.plot(thresholds, nb_r3, color=C_REG3, lw=2.2, linestyle="-",
            label="Regime 3: Within-Fold Selection (In-Fold Pre-filter, $K=4$)", zorder=6)
    ax.plot(thresholds, nb_r2, color=C_REG2, lw=1.9, linestyle=(0, (1.2, 2.2)),
            marker="o", markersize=3.2, markevery=5,
            label="Regime 2: Within-Fold Selection (Cohort Pre-filter, $K=4$)", zorder=7)

    ax.set_xlabel("Threshold Probability ($p_t$)", fontsize=11, labelpad=8)
    ax.set_ylabel("Net Benefit", fontsize=11, labelpad=8)
    ax.set_xlim([0.0, 0.90])
    ax.set_ylim([-0.25, 0.62])
    ax.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=1)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white",
              edgecolor="#cbd5e1", framealpha=0.95)
    fig.tight_layout()
    save_png_and_tif(fig, outdir, "Fig4")
    plt.close(fig)


def plot_fig5_roc(y, p_fib4, p_core, p_reg1, p_reg2, p_reg3, outdir):
    fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", lw=1.3,
            label="Reference Diagonal (AUC = 0.500)", zorder=1)

    if p_fib4 is not None:
        fpr_f4, tpr_f4, _ = roc_curve(y, p_fib4)
        ax.plot(fpr_f4, tpr_f4, color=C_FIB4, linestyle="-.", lw=1.8,
                label=f"A Priori FIB-4 Alone (AUC = {roc_auc_score(y, p_fib4):.3f})", zorder=3)

    fpr_co, tpr_co, _ = roc_curve(y, p_core)
    ax.plot(fpr_co, tpr_co, color=C_CORE, linestyle="-", lw=2.1,
            label=f"Selected Clinical Core: FIB-4 + Sex (AUC = {roc_auc_score(y, p_core):.3f})", zorder=5)

    fpr_r1, tpr_r1, _ = roc_curve(y, p_reg1)
    ax.plot(fpr_r1, tpr_r1, color=C_REG1, linestyle=(0, (5, 2)), lw=2.3,
            label=f"Regime 1: Consensus Hybrid, Fixed $K=4$ (AUC = {roc_auc_score(y, p_reg1):.3f})", zorder=6)

    fpr_r3, tpr_r3, _ = roc_curve(y, p_reg3)
    ax.plot(fpr_r3, tpr_r3, color=C_REG3, linestyle="-", lw=2.0,
            label=f"Regime 3: Within-Fold Selection, In-Fold Pre-filter, $K=4$ (AUC = {roc_auc_score(y, p_reg3):.3f})",
            zorder=4)

    fpr_r2, tpr_r2, _ = roc_curve(y, p_reg2)
    ax.plot(fpr_r2, tpr_r2, color=C_REG2, linestyle=(0, (1.2, 2.2)),
            marker="o", markersize=3.0, markevery=6, lw=1.9,
            label=f"Regime 2: Within-Fold Selection, Cohort Pre-filter, $K=4$ (AUC = {roc_auc_score(y, p_reg2):.3f})",
            zorder=5)

    ax.set_xlabel("1 − Specificity (False Positive Rate)", fontsize=11, labelpad=8)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11, labelpad=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=0)
    ax.legend(loc="lower right", fontsize=8.3, frameon=True, facecolor="white",
              edgecolor="#cbd5e1", framealpha=0.96)
    fig.tight_layout()
    save_png_and_tif(fig, outdir, "Fig5", pad_inches=0.05)
    plt.close(fig)


def plot_s1_fig_all_k(all_k_df, outdir):
    y = all_k_df["y_true"].to_numpy().astype(int)
    p_core = all_k_df["p_clinical_core"].to_numpy().astype(float)
    n = len(y)

    thresholds = np.round(np.arange(0.01, 0.905, 0.01), 2)
    nb_all = (np.sum(y == 1) / n) - (np.sum(y == 0) / n) * (thresholds / (1.0 - thresholds))
    nb_none = np.zeros_like(thresholds)
    nb_core = net_benefit(y, p_core, thresholds)
    g_core, l_core = loess_curve(y, p_core)

    fig, axes = plt.subplots(5, 2, figsize=(7.2, 8.4), dpi=300)
    for row_idx, k in enumerate(range(1, 6)):
        pr1 = all_k_df[f"p_reg1_k{k}"].to_numpy().astype(float)
        pr2 = all_k_df[f"p_reg2_k{k}"].to_numpy().astype(float)
        pr3 = all_k_df[f"p_reg3_k{k}"].to_numpy().astype(float)

        ax_cal = axes[row_idx, 0]
        ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#64748b", lw=0.9,
                    label="Perfect Calibration" if row_idx == 0 else None, zorder=2)
        ax_cal.plot(g_core, l_core, color=C_CORE, lw=1.4, linestyle="-",
                    label="Selected Clinical Core (FIB-4 + Sex)" if row_idx == 0 else None, zorder=4)
        g1, l1 = loess_curve(y, pr1)
        g2, l2 = loess_curve(y, pr2)
        g3, l3 = loess_curve(y, pr3)
        ax_cal.plot(g1, l1, color=C_REG1, lw=1.5, linestyle=(0, (5, 2)),
                    label="Regime 1: Consensus Hybrid" if row_idx == 0 else None, zorder=5)
        ax_cal.plot(g3, l3, color=C_REG3, lw=1.5, linestyle="-",
                    label="Regime 3: In-Fold Pre-filter + L1" if row_idx == 0 else None, zorder=6)
        ax_cal.plot(g2, l2, color=C_REG2, lw=1.3, linestyle=(0, (1.2, 2.2)),
                    marker="o", markersize=2.0, markevery=10,
                    label="Regime 2: Cohort Pre-filter + L1" if row_idx == 0 else None, zorder=7)
        ax_cal.set_xlim(0.0, 1.0)
        ax_cal.set_ylim(0.0, 1.0)
        ax_cal.set_yticks([0.0, 0.5, 1.0])
        ax_cal.set_ylabel(f"$K = {k}$\nObs. Frac.", fontsize=7.8, fontweight="bold")
        if row_idx == 0:
            ax_cal.set_title("A. Flexible LOESS Calibration", fontsize=8.8, fontweight="bold", pad=4)
        if row_idx == 4:
            ax_cal.set_xlabel("Mean OOF Predicted Probability", fontsize=8.0)
        else:
            ax_cal.set_xticklabels([])
        ax_cal.tick_params(labelsize=7.2)
        ax_cal.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=0)

        ax_dca = axes[row_idx, 1]
        ax_dca.axvspan(0.35, 0.65, color="#f1f5f9", alpha=0.9, zorder=0,
                       label="Threshold Range ($p_t = 0.35$–$0.65$)" if row_idx == 0 else None)
        ax_dca.axvline(0.35, color="#94a3b8", linestyle="--", lw=0.7, zorder=1)
        ax_dca.axvline(0.65, color="#94a3b8", linestyle="--", lw=0.7, zorder=1)
        ax_dca.plot(thresholds, nb_none, color="#1e293b", lw=0.9,
                    label="Treat None" if row_idx == 0 else None, zorder=2)
        ax_dca.plot(thresholds, nb_all, linestyle=(0, (3, 3)), color="#64748b", lw=1.0,
                    label="Treat All" if row_idx == 0 else None, zorder=2)
        ax_dca.plot(thresholds, nb_core, color=C_CORE, lw=1.4, linestyle="-", zorder=4)
        ax_dca.plot(thresholds, net_benefit(y, pr1, thresholds), color=C_REG1, lw=1.5,
                    linestyle=(0, (5, 2)), zorder=5)
        ax_dca.plot(thresholds, net_benefit(y, pr3, thresholds), color=C_REG3, lw=1.5,
                    linestyle="-", zorder=6)
        ax_dca.plot(thresholds, net_benefit(y, pr2, thresholds), color=C_REG2, lw=1.3,
                    linestyle=(0, (1.2, 2.2)), marker="o", markersize=2.0, markevery=8, zorder=7)
        ax_dca.set_xlim(0.0, 0.90)
        ax_dca.set_ylim(-0.25, 0.62)
        ax_dca.set_yticks([-0.2, 0.0, 0.3, 0.6])
        ax_dca.set_ylabel("Net Benefit", fontsize=7.8)
        if row_idx == 0:
            ax_dca.set_title("B. Decision Curve Analysis", fontsize=8.8, fontweight="bold", pad=4)
        if row_idx == 4:
            ax_dca.set_xlabel("Threshold Probability ($p_t$)", fontsize=8.0)
        else:
            ax_dca.set_xticklabels([])
        ax_dca.tick_params(labelsize=7.2)
        ax_dca.grid(True, linestyle=":", color="#cbd5e1", alpha=0.7, zorder=1)

    handles_cal, labels_cal = axes[0, 0].get_legend_handles_labels()
    handles_dca, labels_dca = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles_cal + handles_dca,
        labels_cal + labels_dca,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=7.0,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.98,
        columnspacing=0.9,
        handlelength=1.9,
        borderpad=0.35,
        labelspacing=0.3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=0.45, w_pad=0.8)
    save_png_and_tif(fig, outdir, "S1_Fig", pad_inches=0.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Fig 3, Fig 4, Fig 5, and S1 Fig (K = 1-5) from nested_cv_pipeline.py outputs."
    )
    parser.add_argument("--indir", default=".", help="Directory containing oof_predictions_all_k1_to_5.csv.")
    parser.add_argument("--outdir", default=".", help="Directory where PNG and LZW TIFF figures are saved.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    all_k_path = os.path.join(args.indir, "oof_predictions_all_k1_to_5.csv")
    k4_path = os.path.join(args.indir, "oof_predictions.csv")

    if os.path.exists(all_k_path):
        df = pd.read_csv(all_k_path)
        has_all_k = True
    elif os.path.exists(k4_path):
        df = pd.read_csv(k4_path)
        has_all_k = False
    else:
        raise FileNotFoundError(
            f"Could not find oof_predictions_all_k1_to_5.csv or oof_predictions.csv in {args.indir}. "
            "Run nested_cv_pipeline.py first."
        )

    y = df["y_true"].to_numpy().astype(int)
    p_fib4 = df["p_fib4"].to_numpy().astype(float) if "p_fib4" in df.columns else None
    p_core = df["p_clinical_core"].to_numpy().astype(float)
    p_reg1 = df["p_reg1_k4"].to_numpy().astype(float)
    p_reg2 = df["p_reg2_k4"].to_numpy().astype(float)
    p_reg3 = df["p_reg3_k4"].to_numpy().astype(float)

    plot_fig3_calibration(y, p_core, p_reg1, p_reg2, p_reg3, args.outdir)
    plot_fig4_dca(y, p_core, p_reg1, p_reg2, p_reg3, args.outdir)
    plot_fig5_roc(y, p_fib4, p_core, p_reg1, p_reg2, p_reg3, args.outdir)
    if has_all_k:
        plot_s1_fig_all_k(df, args.outdir)
        print("Saved Fig3, Fig4, Fig5, S1_Fig (.png and 300-dpi LZW .tif), and dca_thresholds.csv.")
    else:
        print("Saved Fig3, Fig4, Fig5 (.png and 300-dpi LZW .tif); provide oof_predictions_all_k1_to_5.csv for S1_Fig.")


if __name__ == "__main__":
    main()
