"""Nested cross-validation pipeline: clinical core vs. clinically anchored
radiomic hybrid models (K = 1-5) under three evaluation regimes.

Regime 1: fixed cohort-derived consensus radiomic features (post-selection).
Regime 2: within-fold L1 selection, cohort-level unsupervised pre-filtering.
Regime 3: within-fold L1 selection, training-fold-only pre-filtering.

In every regime and for the clinical core, scaling (RobustScaler), L2 penalty
tuning (inner 5-fold CV) and model fitting use the outer-training fold only.

Usage:
    python nested_cv_pipeline.py --input LBase.txt --outdir results
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

# Keep solver convergence and numerical warnings visible; silence deprecation/fragmentation notices.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*penalty is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SEED = 42
N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 20
N_INNER_SPLITS = 5
N_BOOT = 1000
MAX_K = 5
N_EXPECTED_RADIOMICS = 851

# Column names as they appear in the input table.
# "Cinsiyet" = sex (binary, coded 0/1); "Label" = significant fibrosis (0/1).
SEX_COL = "Cinsiyet"
LABEL_COL = "Label"
ALL_CLINICAL_COLS = [
    "Age", SEX_COL, "AST", "ALT", "PLT", "ALP", "GGT",
    "Tbil", "Dbil", "INR", "PTZ", "Urea", "APRI", "FIB4", LABEL_COL,
]
CLINICAL_CORE = ["FIB4", SEX_COL]
CONSENSUS_RADIOMICS = [
    "wavelet-HLL_glszm_GrayLevelNonUniformityNormalized",
    "wavelet-LLL_firstorder_10Percentile",
    "original_shape_Sphericity",
    "wavelet-LLH_glcm_ClusterProminence",
    "wavelet-LLH_glcm_MCC",
]
C_GRID = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]

REGIMES = ["Regime1_Consensus", "Regime2_CohortPrefilter", "Regime3_InFoldPrefilter"]


# --------------------------------------------------------------------------
# Feature pre-filtering and selection
# --------------------------------------------------------------------------
def prefilter_radiomics(X_df, corr_thr=0.85, var_thr=0.001):
    """Unsupervised filter: min-max variance >= var_thr, then greedy removal of
    features with |Spearman r| > corr_thr (higher-variance feature is kept)."""
    rng = X_df.max() - X_df.min()
    rng[rng == 0] = 1.0
    Xn = (X_df - X_df.min()) / rng
    vars_ = Xn.var()
    keep_var = [c for c in X_df.columns if vars_[c] >= var_thr]
    order_df = pd.DataFrame({"var": vars_[keep_var], "name": keep_var})
    ranked = order_df.sort_values(by=["var", "name"], ascending=[False, True])["name"].tolist()
    corr = X_df[ranked].corr(method="spearman").abs()
    dropped = set()
    for ci in ranked:
        if ci in dropped:
            continue
        for cj in corr.index[corr[ci] > corr_thr].tolist():
            if cj != ci and cj not in dropped:
                dropped.add(cj)
    return [c for c in ranked if c not in dropped]


def l1_entry_order(X_df, y, max_k=MAX_K):
    """Order features by entry along the L1-penalized logistic path
    (features standardized within the data passed in)."""
    Xs = StandardScaler().fit_transform(X_df.values)
    selected = []
    for c_val in np.logspace(-4, 1, 100):
        clf = LogisticRegression(penalty="l1", C=c_val, solver="liblinear",
                                 random_state=SEED, max_iter=500)
        clf.fit(Xs, y)
        coefs = np.abs(clf.coef_.ravel())
        nz = np.where(coefs > 1e-5)[0]
        for idx in nz[np.argsort(-coefs[nz])]:
            fname = X_df.columns[idx]
            if fname not in selected:
                selected.append(fname)
                if len(selected) >= max_k:
                    return selected
    return selected[:max_k]


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------
def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return thresholds[np.argmax(tpr - fpr)]


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOT, seed=SEED):
    np.random.seed(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def hosmer_lemeshow(y_true, y_prob, g=10):
    tmp = pd.DataFrame({"y": y_true, "p": y_prob})
    tmp["q"] = pd.qcut(tmp["p"].rank(method="first"), g, labels=False)
    obs_e = tmp.groupby("q")["y"].sum()
    obs_n = tmp.groupby("q")["y"].count()
    obs_ne = obs_n - obs_e
    exp_e = tmp.groupby("q")["p"].sum()
    exp_ne = obs_n - exp_e
    stat = 0.0
    for q in range(g):
        oe, one = obs_e.get(q, 0), obs_ne.get(q, 0)
        ee, ene = exp_e.get(q, 0), exp_ne.get(q, 0)
        stat += (oe - ee) ** 2 / (ee + 1e-12) + (one - ene) ** 2 / (ene + 1e-12)
    return chi2.sf(stat, df=g - 2)


def calibration_intercept_slope(y_true, y_prob):
    """Calibration-in-the-large (intercept with logit offset) and calibration slope."""
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    lp = np.log(p / (1.0 - p))
    res_int = sm.GLM(y_true, np.ones((len(y_true), 1)),
                     family=sm.families.Binomial(), offset=lp).fit()
    res_slp = sm.GLM(y_true, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(res_int.params[0]), float(res_slp.params[1])


def delong_test(y_true, prob_a, prob_b):
    """Paired DeLong test; returns AUC(b) - AUC(a) and two-sided p value."""
    def _components(y_t, p):
        pos, neg = p[y_t == 1], p[y_t == 0]
        m, n = len(pos), len(neg)
        v10 = np.array([np.mean(neg < x) + 0.5 * np.mean(neg == x) for x in pos])
        v01 = np.array([np.mean(pos > v) + 0.5 * np.mean(pos == v) for v in neg])
        return np.mean(v10), np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n, v10, v01

    auc_a, var_a, v10_a, v01_a = _components(y_true, prob_a)
    auc_b, var_b, v10_b, v01_b = _components(y_true, prob_b)
    m, n = len(v10_a), len(v01_a)
    cov = np.cov(v10_a, v10_b)[0, 1] / m + np.cov(v01_a, v01_b)[0, 1] / n
    diff = auc_b - auc_a
    var_diff = var_a + var_b - 2.0 * cov
    z = diff / np.sqrt(var_diff) if var_diff > 0 else 0.0
    return diff, 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))


def idi_nri(y_true, p_ref, p_new):
    """Point estimates of IDI and continuous (category-free) NRI."""
    ev, nev = (y_true == 1), (y_true == 0)
    idi = (np.mean(p_new[ev]) - np.mean(p_ref[ev])) - (np.mean(p_new[nev]) - np.mean(p_ref[nev]))
    up_ev, dn_ev = np.sum(p_new[ev] > p_ref[ev]), np.sum(p_new[ev] < p_ref[ev])
    up_nev, dn_nev = np.sum(p_new[nev] > p_ref[nev]), np.sum(p_new[nev] < p_ref[nev])
    nri = (up_ev - dn_ev) / np.sum(ev) + (dn_nev - up_nev) / np.sum(nev)
    return idi, nri


def bootstrap_idi_nri_ci(y_true, p_ref, p_new, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap 95% CIs (patients resampled) for IDI and continuous NRI.
    Reported instead of asymptotic p values, which are not valid for nested models."""
    rs = np.random.RandomState(seed)
    n = len(y_true)
    idis, nris = [], []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        i, r = idi_nri(y_true[idx], p_ref[idx], p_new[idx])
        idis.append(i)
        nris.append(r)
    return (np.percentile(idis, 2.5), np.percentile(idis, 97.5),
            np.percentile(nris, 2.5), np.percentile(nris, 97.5))


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def tune_l2_c(X_tr_df, y_tr, cont_cols, outer_idx):
    """Select the L2 penalty C by mean inner-CV AUC, using the outer-training fold only.
    The inner split depends only on outer_idx, so every model evaluated in the same
    outer fold (clinical core and all regimes) sees identical inner folds."""
    inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True,
                               random_state=SEED + outer_idx)
    scores = {c: [] for c in C_GRID}
    for tr_in, val_in in inner_cv.split(X_tr_df, y_tr):
        X_in_tr = X_tr_df.iloc[tr_in].copy()
        X_in_val = X_tr_df.iloc[val_in].copy()
        sc = RobustScaler()
        X_in_tr[cont_cols] = sc.fit_transform(X_in_tr[cont_cols])
        X_in_val[cont_cols] = sc.transform(X_in_val[cont_cols])
        for c_val in C_GRID:
            clf = LogisticRegression(penalty="l2", C=c_val, solver="liblinear", random_state=SEED)
            clf.fit(X_in_tr.values, y_tr[tr_in])
            p_val = clf.predict_proba(X_in_val.values)[:, 1]
            scores[c_val].append(roc_auc_score(y_tr[val_in], p_val))
    mean_scores = {c: np.mean(v) for c, v in scores.items()}
    return max(mean_scores, key=mean_scores.get)


def fit_predict(df, tr, te, ytr, feats, outer_idx):
    """Tune C, scale and fit on the outer-training fold; predict both folds."""
    cont = [c for c in feats if c != SEX_COL]
    c_tuned = tune_l2_c(df.iloc[tr][feats], ytr, cont, outer_idx)
    Xtr = df.iloc[tr][feats].copy()
    Xte = df.iloc[te][feats].copy()
    sc = RobustScaler()
    Xtr[cont] = sc.fit_transform(Xtr[cont])
    Xte[cont] = sc.transform(Xte[cont])
    clf = LogisticRegression(penalty="l2", C=c_tuned, solver="liblinear",
                             random_state=SEED).fit(Xtr.values, ytr)
    return clf.predict_proba(Xtr.values)[:, 1], clf.predict_proba(Xte.values)[:, 1]


def summary_row(name, k, y, p_pool, rep_probs, tr_auc=np.nan, sens=np.nan, spec=np.nan,
                thr_mean=np.nan, thr_sd=np.nan):
    ci_l, ci_u = bootstrap_auc_ci(y, p_pool)
    rep_aucs = [roc_auc_score(y, rep_probs[r]) for r in range(rep_probs.shape[0])]
    cal_int, cal_slp = calibration_intercept_slope(y, p_pool)
    return {
        "Regime": name, "K": k,
        "Train_AUC": tr_auc,
        "Pooled_OOF_AUC": roc_auc_score(y, p_pool), "CI_95": f"{ci_l:.3f}-{ci_u:.3f}",
        "Repeat_AUC_Mean": np.mean(rep_aucs), "Repeat_AUC_SD": np.std(rep_aucs),
        "Sens_pct": sens, "Spec_pct": spec, "Bal_Acc_pct": 0.5 * (sens + spec),
        "Thr_Mean": thr_mean, "Thr_SD": thr_sd,
        "Brier": brier_score_loss(y, p_pool),
        "Cal_Intercept": cal_int, "Cal_Slope": cal_slp,
        "HL_p": hosmer_lemeshow(y, p_pool),
        "Pred_Min": p_pool.min(), "Pred_Max": p_pool.max(), "Pred_Mean": p_pool.mean(),
    }


def run_nested_cv(df, y, radiomic_cols):
    n_samples = len(y)
    ks = range(1, MAX_K + 1)
    # Regime 2 only: unsupervised pre-filter fitted once on the whole cohort.
    global_pool = prefilter_radiomics(df[radiomic_cols])
    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS,
                                   random_state=SEED)

    oof_fib4 = np.zeros((N_OUTER_REPEATS, n_samples))
    oof_clin = np.zeros((N_OUTER_REPEATS, n_samples))
    tr_aucs_clin = []
    oof_probs = {r: {k: np.zeros((N_OUTER_REPEATS, n_samples)) for k in ks} for r in REGIMES}
    oof_preds = {r: {k: np.zeros((N_OUTER_REPEATS, n_samples)) for k in ks} for r in REGIMES}
    tr_aucs = {r: {k: [] for k in ks} for r in REGIMES}
    thrs = {r: {k: [] for k in ks} for r in REGIMES}

    for outer_idx, (tr, te) in enumerate(rskf.split(df, y)):
        rep = outer_idx // N_OUTER_SPLITS
        ytr = y[tr]

        # A priori FIB-4 baseline and selected clinical core in the unified pipeline.
        _, p_te_fib4 = fit_predict(df, tr, te, ytr, ["FIB4"], outer_idx)
        oof_fib4[rep, te] = p_te_fib4
        p_tr, p_te = fit_predict(df, tr, te, ytr, CLINICAL_CORE, outer_idx)
        oof_clin[rep, te] = p_te
        tr_aucs_clin.append(roc_auc_score(ytr, p_tr))

        # Within-fold supervised ranking (Regimes 2 and 3).
        top_r2 = l1_entry_order(df.iloc[tr][global_pool], ytr)
        infold_pool = prefilter_radiomics(df.iloc[tr][radiomic_cols])
        top_r3 = l1_entry_order(df.iloc[tr][infold_pool], ytr)

        for k in ks:
            f_sets = {
                "Regime1_Consensus": CLINICAL_CORE + CONSENSUS_RADIOMICS[:k],
                "Regime2_CohortPrefilter": CLINICAL_CORE + top_r2[:k],
                "Regime3_InFoldPrefilter": CLINICAL_CORE + top_r3[:k],
            }
            for r_name, feats in f_sets.items():
                # C is tuned separately for each regime on its own feature set.
                p_tr, p_te = fit_predict(df, tr, te, ytr, feats, outer_idx)
                oof_probs[r_name][k][rep, te] = p_te
                tr_aucs[r_name][k].append(roc_auc_score(ytr, p_tr))
                t_opt = youden_threshold(ytr, p_tr)  # threshold from training fold only
                thrs[r_name][k].append(t_opt)
                oof_preds[r_name][k][rep, te] = (p_te >= t_opt).astype(int)

    rows = [summary_row("ClinicalCore", 0, y, np.mean(oof_clin, axis=0), oof_clin,
                        tr_auc=np.mean(tr_aucs_clin))]
    for r_name in REGIMES:
        for k in ks:
            pred_cons = (np.mean(oof_preds[r_name][k], axis=0) >= 0.50).astype(int)
            sens = ((pred_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
            spec = ((pred_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
            rows.append(summary_row(
                r_name, k, y, np.mean(oof_probs[r_name][k], axis=0), oof_probs[r_name][k],
                tr_auc=np.mean(tr_aucs[r_name][k]), sens=sens, spec=spec,
                thr_mean=np.mean(thrs[r_name][k]), thr_sd=np.std(thrs[r_name][k])))

    oof_df = pd.DataFrame({
        "patient_id": np.arange(n_samples),
        "y_true": y,
        "p_fib4": np.mean(oof_fib4, axis=0),
        "p_clinical_core": np.mean(oof_clin, axis=0),
        "p_reg1_k4": np.mean(oof_probs["Regime1_Consensus"][4], axis=0),
        "p_reg2_k4": np.mean(oof_probs["Regime2_CohortPrefilter"][4], axis=0),
        "p_reg3_k4": np.mean(oof_probs["Regime3_InFoldPrefilter"][4], axis=0),
    })

    all_k_cols = {
        "patient_id": np.arange(n_samples),
        "y_true": y,
        "p_fib4": np.mean(oof_fib4, axis=0),
        "p_clinical_core": np.mean(oof_clin, axis=0),
    }
    for k in ks:
        all_k_cols[f"p_reg1_k{k}"] = np.mean(oof_probs["Regime1_Consensus"][k], axis=0)
        all_k_cols[f"p_reg2_k{k}"] = np.mean(oof_probs["Regime2_CohortPrefilter"][k], axis=0)
        all_k_cols[f"p_reg3_k{k}"] = np.mean(oof_probs["Regime3_InFoldPrefilter"][k], axis=0)
    oof_all_k_df = pd.DataFrame(all_k_cols)

    return pd.DataFrame(rows), oof_df, oof_all_k_df


def incremental_value(y, oof_df):
    """Hybrid (K = 4) vs. clinical core, from out-of-fold predictions, for each regime."""
    p_core = oof_df["p_clinical_core"].values
    rows = []
    for label, col in [("Regime1_Consensus", "p_reg1_k4"),
                       ("Regime2_CohortPrefilter", "p_reg2_k4"),
                       ("Regime3_InFoldPrefilter", "p_reg3_k4")]:
        p_new = oof_df[col].values
        d_auc, p_delong = delong_test(y, p_core, p_new)
        idi, nri = idi_nri(y, p_core, p_new)
        idi_l, idi_u, nri_l, nri_u = bootstrap_idi_nri_ci(y, p_core, p_new)
        rows.append({
            "Regime": label, "K": 4,
            "Delta_OOF_AUC": d_auc, "DeLong_p": p_delong,
            "IDI": idi, "IDI_CI_95": f"{idi_l:.4f} to {idi_u:.4f}",
            "cNRI": nri, "cNRI_CI_95": f"{nri_l:.3f} to {nri_u:.3f}",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Nested cross-validation pipeline for CT liver fibrosis.")
    parser.add_argument("--input", default="LBase.txt", help="Tab-delimited input table.")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    y = df[LABEL_COL].to_numpy().astype(int)
    df["FIB4"] = (df["Age"] * df["AST"]) / (df["PLT"] * np.sqrt(df["ALT"]))
    radiomic_cols = [c for c in df.columns if c not in ALL_CLINICAL_COLS]

    if len(radiomic_cols) != N_EXPECTED_RADIOMICS:
        raise ValueError(
            f"Expected {N_EXPECTED_RADIOMICS} radiomic columns, found {len(radiomic_cols)}. "
            "Check the input table for identifier or diagnostic columns.")
    missing = [c for c in CONSENSUS_RADIOMICS if c not in radiomic_cols]
    if missing:
        raise ValueError(f"Consensus features not found in input: {missing}")
    if df[radiomic_cols + CLINICAL_CORE].isna().any().any():
        raise ValueError("Missing values found in model input columns.")

    summary_df, oof_df, oof_all_k_df = run_nested_cv(df, y, radiomic_cols)
    incr_df = incremental_value(y, oof_df)

    fmt = lambda v: f"{v:.3f}"
    print(summary_df.to_string(index=False, float_format=fmt))
    print("\nIncremental value over the clinical core (out-of-fold, K = 4):")
    print(incr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary_df.to_csv(os.path.join(args.outdir, "nested_cv_summary.csv"), index=False)
    oof_df.to_csv(os.path.join(args.outdir, "oof_predictions.csv"), index=False)
    oof_all_k_df.to_csv(os.path.join(args.outdir, "oof_predictions_all_k1_to_5.csv"), index=False)
    incr_df.to_csv(os.path.join(args.outdir, "incremental_value.csv"), index=False)


if __name__ == "__main__":
    main()
