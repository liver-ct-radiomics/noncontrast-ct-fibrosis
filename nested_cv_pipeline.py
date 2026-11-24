"""Unified nested cross-validation modeling and evaluation pipeline for
non-contrast CT liver fibrosis staging (reproduces Tables 2-5, out-of-fold
probability tables for Figs 3-5 and S1 Fig, full-cohort descriptive LRT, and
out-of-fold DeLong / bootstrap IDI / continuous NRI comparisons).

Models evaluated using 5-fold stratified nested cross-validation repeated 20 times (identical outer splits):
  - Table 2: A priori serum indices (APRI, FIB-4) and sequential clinical models
    (Clinical models 2-5; FIB-4 + sex carried forward as the clinical core).
  - Table 3: Standalone radiomic-only models (2-5 parameters; training-fold
    unsupervised pre-filtering and L1 path ranking).
  - Table 4: Unanchored joint clinical + radiomic models (3-7 parameters;
    training-fold pre-filtering, plus cohort-prefiltered sensitivity check).
  - Table 5: Clinically anchored hybrid models (K = 1-5 radiomic additions):
      * Regime 1: fixed cohort-derived consensus radiomic features (post-selection).
      * Regime 2: within-fold L1 selection, cohort-level unsupervised pre-filtering.
      * Regime 3: within-fold L1 selection, training-fold-only pre-filtering.
  - Incremental value: Descriptive full-cohort unpenalized likelihood-ratio test
    (LRT) and out-of-fold paired DeLong test, IDI, and continuous NRI at K = 4.

In every model, scaling (RobustScaler), L2 penalty tuning (inner 5-fold CV),
model fitting, and Youden threshold selection use the outer-training fold only.

Usage:
    python nested_cv_pipeline.py --input LBase.txt --outdir results
"""
import argparse
import os
import warnings
from collections import Counter

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
AST_ULN = 35.0

# Column names as they appear in the input table.
# "Cinsiyet" = sex (binary, coded 0/1); "Label" = study-defined significant fibrosis (0/1).
SEX_COL = "Cinsiyet"
LABEL_COL = "Label"
ALL_CLINICAL_COLS = [
    "Age", SEX_COL, "AST", "ALT", "PLT", "ALP", "GGT",
    "Tbil", "Dbil", "INR", "PTZ", "Urea", "APRI", "FIB4", LABEL_COL,
]
NON_COLLINEAR_CLINICAL = [
    "FIB4", SEX_COL, "ALP", "GGT", "Tbil", "Dbil", "INR", "PTZ", "Urea",
]
CLINICAL_CORE = ["FIB4", SEX_COL]
CLINICAL_MODEL_SPECS = [
    ("APRI", ["APRI"], "(AST / AST_ULN) / PLT * 100"),
    ("FIB-4", ["FIB4"], "(Age * AST) / (PLT * sqrt(ALT))"),
    ("Clinical model 2", ["FIB4", SEX_COL], "FIB-4 + Sex"),
    ("Clinical model 3", ["FIB4", SEX_COL, "ALP"], "Clin 2 + ALP"),
    ("Clinical model 4", ["FIB4", SEX_COL, "ALP", "GGT"], "Clin 3 + GGT"),
    ("Clinical model 5", ["FIB4", SEX_COL, "ALP", "GGT", "Tbil"], "Clin 4 + TBil"),
]
RADIOMIC_ONLY_SIZES = [2, 3, 4, 5]
UNANCHORED_SIZES = [3, 4, 5, 6, 7]
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


def l1_entry_order(X_df, y, max_k=MAX_K, unscaled_cols=()):
    """Order features by entry along the L1-penalized logistic path
    (continuous features standardized within the data passed in; binary columns
    listed in unscaled_cols are left unscaled)."""
    if unscaled_cols:
        X_work = X_df.copy()
        cont = [c for c in X_work.columns if c not in unscaled_cols]
        X_work[cont] = StandardScaler().fit_transform(X_work[cont].values)
        Xs = X_work.values
    else:
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


def combined_repeat_auc_ci(y_true, rep_probs, n_boot=N_BOOT, seed=SEED):
    """Approximate 95% CI combining mean within-repeat bootstrap variance (W_bar)
    and between-repeat variance (B) across R repeats: T = W_bar + (1 + 1/R) * B."""
    r_count, n = rep_probs.shape
    p_pool = np.mean(rep_probs, axis=0)
    auc_pool = roc_auc_score(y_true, p_pool)
    rep_aucs = np.array([roc_auc_score(y_true, rep_probs[r]) for r in range(r_count)])
    b_var = np.var(rep_aucs, ddof=1)
    w_vars = []
    for r in range(r_count):
        rs = np.random.RandomState(seed + r)
        b_aucs = []
        for _ in range(n_boot):
            idx = rs.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            b_aucs.append(roc_auc_score(y_true[idx], rep_probs[r, idx]))
        w_vars.append(np.var(b_aucs, ddof=1))
    total_var = float(np.mean(w_vars)) + (1.0 + 1.0 / r_count) * float(b_var)
    half_width = 1.96 * np.sqrt(total_var)
    return auc_pool - half_width, auc_pool + half_width


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


def full_cohort_lrt(df, y):
    """Descriptive full-cohort unpenalized maximum-likelihood likelihood-ratio test
    comparing the selected clinical core (FIB-4 + sex + intercept) against the
    four-feature consensus hybrid model (K = 4; df = 4)."""
    feats_reduced = CLINICAL_CORE
    feats_full = CLINICAL_CORE + CONSENSUS_RADIOMICS[:4]
    m_red = sm.Logit(y, sm.add_constant(df[feats_reduced])).fit(disp=False)
    m_full = sm.Logit(y, sm.add_constant(df[feats_full])).fit(disp=False)
    stat = float(2.0 * (m_full.llf - m_red.llf))
    df_diff = len(feats_full) - len(feats_reduced)
    p_val = float(chi2.sf(stat, df=df_diff))
    return pd.DataFrame([{
        "Reduced_Model": "FIB-4 + Sex + Intercept",
        "Full_Model": "FIB-4 + Sex + 4 Consensus Radiomics + Intercept",
        "LL_Reduced": float(m_red.llf),
        "LL_Full": float(m_full.llf),
        "Chi2_Stat": stat,
        "df": df_diff,
        "LRT_p": p_val,
    }])


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def tune_l2_c(X_tr_df, y_tr, cont_cols, outer_idx):
    """Select the L2 penalty C by mean inner-CV AUC, using the outer-training fold only.
    The inner split depends only on outer_idx, so every model evaluated in the same
    outer fold sees identical inner folds."""
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


def fit_predict(df, tr, te, ytr, feats, outer_idx, return_c=False):
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
    p_tr = clf.predict_proba(Xtr.values)[:, 1]
    p_te = clf.predict_proba(Xte.values)[:, 1]
    if return_c:
        return p_tr, p_te, c_tuned
    return p_tr, p_te


def format_modal_c(c_values):
    counts = Counter(c_values).most_common()
    top_n = counts[0][1]
    modes = [val for val, cnt in counts if cnt == top_n]
    if len(modes) == 1:
        return f"{modes[0]:.2f}"
    return " / ".join(f"{m:.2f}" for m in modes) + " (co-modal)"


def summary_row(name, k, y, p_pool, rep_probs, tr_auc=np.nan, sens=np.nan, spec=np.nan,
                thr_mean=np.nan, thr_sd=np.nan):
    ci_l, ci_u = bootstrap_auc_ci(y, p_pool)
    comb_l, comb_u = combined_repeat_auc_ci(y, rep_probs)
    rep_aucs = [roc_auc_score(y, rep_probs[r]) for r in range(rep_probs.shape[0])]
    cal_int, cal_slp = calibration_intercept_slope(y, p_pool)
    return {
        "Regime": name, "K": k,
        "Train_AUC": tr_auc,
        "Pooled_OOF_AUC": roc_auc_score(y, p_pool),
        "CI_95": f"{ci_l:.3f}-{ci_u:.3f}",
        "Combined_Var_CI_95": f"{comb_l:.3f}-{comb_u:.3f}",
        "Repeat_AUC_Mean": np.mean(rep_aucs), "Repeat_AUC_SD": np.std(rep_aucs),
        "Sens_pct": sens, "Spec_pct": spec, "Bal_Acc_pct": 0.5 * (sens + spec),
        "Thr_Mean": thr_mean, "Thr_SD": thr_sd,
        "Brier": brier_score_loss(y, p_pool),
        "Cal_Intercept": cal_int, "Cal_Slope": cal_slp,
        "HL_p": hosmer_lemeshow(y, p_pool),
        "Pred_Min": p_pool.min(), "Pred_Max": p_pool.max(), "Pred_Mean": p_pool.mean(),
    }


def evaluate_all_tracks(df, y, radiomic_cols):
    """Run all nested CV modeling tracks (Tables 2, 3, 4, and 5) across the same
    100 outer train-test splits (5-fold stratified CV repeated 20 times)."""
    n_samples = len(y)
    ks = range(1, MAX_K + 1)
    # Cohort-level unsupervised pre-filter (used in Regime 2 and Table 4 footnote check).
    global_pool = prefilter_radiomics(df[radiomic_cols])
    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS,
                                   random_state=SEED)

    # --- Track 1 (Table 2): A priori serum indices & sequential clinical models ---
    oof_t2 = {nm: np.zeros((N_OUTER_REPEATS, n_samples)) for nm, _, _ in CLINICAL_MODEL_SPECS}
    pred_t2 = {nm: np.zeros((N_OUTER_REPEATS, n_samples)) for nm, _, _ in CLINICAL_MODEL_SPECS}
    tr_t2 = {nm: [] for nm, _, _ in CLINICAL_MODEL_SPECS}
    thr_t2 = {nm: [] for nm, _, _ in CLINICAL_MODEL_SPECS}

    # --- Track 2 (Table 3): Standalone radiomic-only models (2-5 parameters) ---
    oof_t3 = {s: np.zeros((N_OUTER_REPEATS, n_samples)) for s in RADIOMIC_ONLY_SIZES}
    pred_t3 = {s: np.zeros((N_OUTER_REPEATS, n_samples)) for s in RADIOMIC_ONLY_SIZES}
    tr_t3 = {s: [] for s in RADIOMIC_ONLY_SIZES}
    thr_t3 = {s: [] for s in RADIOMIC_ONLY_SIZES}
    infold_pool_sizes = []

    # --- Track 3 (Table 4): Unanchored joint hybrid models (3-7 parameters) ---
    unanch_modes = ["InFoldPrefilter", "CohortPrefilter"]
    oof_t4 = {m: {s: np.zeros((N_OUTER_REPEATS, n_samples)) for s in UNANCHORED_SIZES} for m in unanch_modes}
    pred_t4 = {m: {s: np.zeros((N_OUTER_REPEATS, n_samples)) for s in UNANCHORED_SIZES} for m in unanch_modes}
    tr_t4 = {m: {s: [] for s in UNANCHORED_SIZES} for m in unanch_modes}
    thr_t4 = {m: {s: [] for s in UNANCHORED_SIZES} for m in unanch_modes}
    fib4_omitted_t4 = {m: {s: 0 for s in UNANCHORED_SIZES} for m in unanch_modes}
    feat_counts_t4 = {m: {s: Counter() for s in UNANCHORED_SIZES} for m in unanch_modes}

    # --- Track 4 (Table 5): Clinically anchored hybrid models (Regimes 1-3, K = 1-5) ---
    oof_probs = {r: {k: np.zeros((N_OUTER_REPEATS, n_samples)) for k in ks} for r in REGIMES}
    oof_preds = {r: {k: np.zeros((N_OUTER_REPEATS, n_samples)) for k in ks} for r in REGIMES}
    tr_aucs = {r: {k: [] for k in ks} for r in REGIMES}
    thrs = {r: {k: [] for k in ks} for r in REGIMES}
    tuned_cs = {r: {k: [] for k in ks} for r in REGIMES}
    freq_top5_r2 = Counter()
    freq_top5_r3 = Counter()

    for outer_idx, (tr, te) in enumerate(rskf.split(df, y)):
        rep = outer_idx // N_OUTER_SPLITS
        ytr = y[tr]

        # 1. Clinical models (Table 2)
        for nm, feats, _ in CLINICAL_MODEL_SPECS:
            p_tr, p_te = fit_predict(df, tr, te, ytr, feats, outer_idx)
            oof_t2[nm][rep, te] = p_te
            tr_t2[nm].append(roc_auc_score(ytr, p_tr))
            t_opt = youden_threshold(ytr, p_tr)
            thr_t2[nm].append(t_opt)
            pred_t2[nm][rep, te] = (p_te >= t_opt).astype(int)

        # Unsupervised pre-filtering and within-fold radiomic L1 ranking
        top_r2 = l1_entry_order(df.iloc[tr][global_pool], ytr, max_k=MAX_K)
        infold_pool = prefilter_radiomics(df.iloc[tr][radiomic_cols])
        infold_pool_sizes.append(len(infold_pool))
        top_r3 = l1_entry_order(df.iloc[tr][infold_pool], ytr, max_k=MAX_K)
        freq_top5_r2.update(top_r2)
        freq_top5_r3.update(top_r3)

        # 2. Radiomic-only models (Table 3; training-fold pre-filtering)
        for s in RADIOMIC_ONLY_SIZES:
            feats_r = top_r3[:s]
            p_tr, p_te = fit_predict(df, tr, te, ytr, feats_r, outer_idx)
            oof_t3[s][rep, te] = p_te
            tr_t3[s].append(roc_auc_score(ytr, p_tr))
            t_opt = youden_threshold(ytr, p_tr)
            thr_t3[s].append(t_opt)
            pred_t3[s][rep, te] = (p_te >= t_opt).astype(int)

        # 3. Unanchored joint hybrid models (Table 4; training-fold + cohort check)
        for m_name, r_pool in [("InFoldPrefilter", infold_pool), ("CohortPrefilter", global_pool)]:
            joint_cols = NON_COLLINEAR_CLINICAL + r_pool
            ranked_joint = l1_entry_order(
                df.iloc[tr][joint_cols], ytr, max_k=max(UNANCHORED_SIZES), unscaled_cols=(SEX_COL,)
            )
            for s in UNANCHORED_SIZES:
                feats_u = ranked_joint[:s]
                if "FIB4" not in feats_u:
                    fib4_omitted_t4[m_name][s] += 1
                feat_counts_t4[m_name][s].update(feats_u)
                p_tr, p_te = fit_predict(df, tr, te, ytr, feats_u, outer_idx)
                oof_t4[m_name][s][rep, te] = p_te
                tr_t4[m_name][s].append(roc_auc_score(ytr, p_tr))
                t_opt = youden_threshold(ytr, p_tr)
                thr_t4[m_name][s].append(t_opt)
                pred_t4[m_name][s][rep, te] = (p_te >= t_opt).astype(int)

        # 4. Clinically anchored hybrid models (Table 5; Regimes 1-3, K = 1-5)
        for k in ks:
            f_sets = {
                "Regime1_Consensus": CLINICAL_CORE + CONSENSUS_RADIOMICS[:k],
                "Regime2_CohortPrefilter": CLINICAL_CORE + top_r2[:k],
                "Regime3_InFoldPrefilter": CLINICAL_CORE + top_r3[:k],
            }
            for r_name, feats in f_sets.items():
                p_tr, p_te, c_sel = fit_predict(df, tr, te, ytr, feats, outer_idx, return_c=True)
                oof_probs[r_name][k][rep, te] = p_te
                tr_aucs[r_name][k].append(roc_auc_score(ytr, p_tr))
                tuned_cs[r_name][k].append(c_sel)
                t_opt = youden_threshold(ytr, p_tr)
                thrs[r_name][k].append(t_opt)
                oof_preds[r_name][k][rep, te] = (p_te >= t_opt).astype(int)

    # Assemble Table 2 DataFrame
    t2_rows = []
    for nm, feats, defn in CLINICAL_MODEL_SPECS:
        pred_cons = (np.mean(pred_t2[nm], axis=0) >= 0.50).astype(int)
        sens = ((pred_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
        spec = ((pred_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
        row = summary_row(
            nm, len(feats), y, np.mean(oof_t2[nm], axis=0), oof_t2[nm],
            tr_auc=np.mean(tr_t2[nm]), sens=sens, spec=spec,
            thr_mean=np.mean(thr_t2[nm]), thr_sd=np.std(thr_t2[nm]),
        )
        row["Definition"] = defn
        t2_rows.append(row)
    table2_df = pd.DataFrame(t2_rows)

    # Assemble Table 3 DataFrame
    t3_rows = []
    for s in RADIOMIC_ONLY_SIZES:
        pred_cons = (np.mean(pred_t3[s], axis=0) >= 0.50).astype(int)
        sens = ((pred_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
        spec = ((pred_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
        row = summary_row(
            "RadiomicsOnly_InFoldPrefilter", s, y, np.mean(oof_t3[s], axis=0), oof_t3[s],
            tr_auc=np.mean(tr_t3[s]), sens=sens, spec=spec,
            thr_mean=np.mean(thr_t3[s]), thr_sd=np.std(thr_t3[s]),
        )
        row["Cohort_Pool_Size"] = len(global_pool)
        row["InFold_Pool_Mean"] = float(np.mean(infold_pool_sizes))
        row["InFold_Pool_Min"] = int(np.min(infold_pool_sizes))
        row["InFold_Pool_Max"] = int(np.max(infold_pool_sizes))
        t3_rows.append(row)
    table3_df = pd.DataFrame(t3_rows)

    # Assemble Table 4 DataFrame (primary InFoldPrefilter + CohortPrefilter footnote check)
    t4_rows = []
    for m_name in unanch_modes:
        for s in UNANCHORED_SIZES:
            pred_cons = (np.mean(pred_t4[m_name][s], axis=0) >= 0.50).astype(int)
            sens = ((pred_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
            spec = ((pred_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
            row = summary_row(
                f"Unanchored_{m_name}", s, y, np.mean(oof_t4[m_name][s], axis=0), oof_t4[m_name][s],
                tr_auc=np.mean(tr_t4[m_name][s]), sens=sens, spec=spec,
                thr_mean=np.mean(thr_t4[m_name][s]), thr_sd=np.std(thr_t4[m_name][s]),
            )
            row["FIB4_Omitted_Pct"] = fib4_omitted_t4[m_name][s]
            row["Features_Selected_gt50pct"] = sum(
                1 for cnt in feat_counts_t4[m_name][s].values() if cnt > 50
            )
            t4_rows.append(row)
    table4_df = pd.DataFrame(t4_rows)

    # Assemble Table 5 (nested_cv_summary.csv) DataFrame
    oof_clin = oof_t2["Clinical model 2"]
    oof_fib4 = oof_t2["FIB-4"]
    pred_clin_cons = (np.mean(pred_t2["Clinical model 2"], axis=0) >= 0.50).astype(int)
    sens_clin = ((pred_clin_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
    spec_clin = ((pred_clin_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
    clin_row = summary_row(
        "ClinicalCore", 0, y, np.mean(oof_clin, axis=0), oof_clin,
        tr_auc=np.mean(tr_t2["Clinical model 2"]), sens=sens_clin, spec=spec_clin,
        thr_mean=np.mean(thr_t2["Clinical model 2"]), thr_sd=np.std(thr_t2["Clinical model 2"]),
    )
    clin_row["Modal_C"] = ""
    rows = [clin_row]
    for r_name in REGIMES:
        for k in ks:
            pred_cons = (np.mean(oof_preds[r_name][k], axis=0) >= 0.50).astype(int)
            sens = ((pred_cons == 1) & (y == 1)).sum() / np.sum(y == 1) * 100.0
            spec = ((pred_cons == 0) & (y == 0)).sum() / np.sum(y == 0) * 100.0
            row = summary_row(
                r_name, k, y, np.mean(oof_probs[r_name][k], axis=0), oof_probs[r_name][k],
                tr_auc=np.mean(tr_aucs[r_name][k]), sens=sens, spec=spec,
                thr_mean=np.mean(thrs[r_name][k]), thr_sd=np.std(thrs[r_name][k]),
            )
            row["Modal_C"] = format_modal_c(tuned_cs[r_name][k])
            rows.append(row)
    summary_df = pd.DataFrame(rows)

    # Top-5 radiomic selection stability across 100 outer splits (Regime 2 vs. Regime 3 / Table 3)
    all_feats_seen = sorted(
        set(freq_top5_r2.keys()) | set(freq_top5_r3.keys()),
        key=lambda f: -(freq_top5_r3[f] + freq_top5_r2[f]),
    )
    freq_df = pd.DataFrame([
        {
            "Feature": f,
            "In_Consensus_Top5": f in CONSENSUS_RADIOMICS,
            "Top5_Freq_CohortPrefilter_Pct": freq_top5_r2[f],
            "Top5_Freq_InFoldPrefilter_Pct": freq_top5_r3[f],
        }
        for f in all_feats_seen
    ])

    # Out-of-fold probability tables consumed by plot_figures.py
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

    return table2_df, table3_df, table4_df, summary_df, freq_df, oof_df, oof_all_k_df


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
    parser = argparse.ArgumentParser(description="Unified nested cross-validation pipeline for CT liver fibrosis.")
    parser.add_argument("--input", default="LBase.txt", help="Tab-delimited input table.")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    y = df[LABEL_COL].to_numpy().astype(int)
    df["APRI"] = (df["AST"] / AST_ULN) * 100.0 / df["PLT"]
    df["FIB4"] = (df["Age"] * df["AST"]) / (df["PLT"] * np.sqrt(df["ALT"]))
    radiomic_cols = [c for c in df.columns if c not in ALL_CLINICAL_COLS]

    if len(radiomic_cols) != N_EXPECTED_RADIOMICS:
        raise ValueError(
            f"Expected {N_EXPECTED_RADIOMICS} radiomic columns, found {len(radiomic_cols)}. "
            "Check the input table for identifier or diagnostic columns.")
    missing = [c for c in CONSENSUS_RADIOMICS if c not in radiomic_cols]
    if missing:
        raise ValueError(f"Consensus features not found in input: {missing}")
    req_clin = ["Age", SEX_COL, "AST", "ALT", "PLT", "ALP", "GGT", "Tbil", "Dbil", "INR", "PTZ", "Urea"]
    if df[radiomic_cols + req_clin].isna().any().any():
        raise ValueError("Missing values found in model input columns.")

    (
        table2_df,
        table3_df,
        table4_df,
        summary_df,
        freq_df,
        oof_df,
        oof_all_k_df,
    ) = evaluate_all_tracks(df, y, radiomic_cols)
    lrt_df = full_cohort_lrt(df, y)
    incr_df = incremental_value(y, oof_df)

    fmt = lambda v: f"{v:.3f}"
    print("=== Table 2: Clinical models ===")
    print(table2_df.to_string(index=False, float_format=fmt))
    print("\n=== Table 3: Radiomic-only models (training-fold pre-filtering) ===")
    print(table3_df.to_string(index=False, float_format=fmt))
    print("\n=== Table 4: Unanchored hybrid models ===")
    print(table4_df.to_string(index=False, float_format=fmt))
    print("\n=== Table 5: Clinically anchored hybrid models (Regimes 1-3, K = 1-5) ===")
    print(summary_df.to_string(index=False, float_format=fmt))
    print("\n=== Descriptive full-cohort unpenalized LRT (K = 4 vs. clinical core) ===")
    print(lrt_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Out-of-fold incremental value over the clinical core (K = 4) ===")
    print(incr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    table2_df.to_csv(os.path.join(args.outdir, "table2_clinical_models.csv"), index=False)
    table3_df.to_csv(os.path.join(args.outdir, "table3_radiomics_only.csv"), index=False)
    table4_df.to_csv(os.path.join(args.outdir, "table4_unanchored_hybrid.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "nested_cv_summary.csv"), index=False)
    freq_df.to_csv(os.path.join(args.outdir, "radiomic_top5_selection_frequencies.csv"), index=False)
    lrt_df.to_csv(os.path.join(args.outdir, "full_cohort_lrt.csv"), index=False)
    oof_df.to_csv(os.path.join(args.outdir, "oof_predictions.csv"), index=False)
    oof_all_k_df.to_csv(os.path.join(args.outdir, "oof_predictions_all_k1_to_5.csv"), index=False)
    incr_df.to_csv(os.path.join(args.outdir, "incremental_value.csv"), index=False)


if __name__ == "__main__":
    main()
