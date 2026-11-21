"""Additional fully nested sensitivity analysis in which clinical expansion
size (m = 0-4 beyond FIB-4), radiomic subset size (K = 0-5), and the L2 penalty
C (0.01-10.0) are jointly selected by 5-fold inner cross-validation under both
the one-standard-error (1-SE) rule and the maximum inner-AUC rule.

Usage:
    python inner_loop_sensitivity.py --input LBase.txt --outdir .
"""
import argparse
from collections import Counter
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from nested_cv_pipeline import (
    ALL_CLINICAL_COLS,
    C_GRID,
    LABEL_COL,
    N_BOOT,
    N_INNER_SPLITS,
    N_OUTER_REPEATS,
    N_OUTER_SPLITS,
    SEED,
    SEX_COL,
    bootstrap_auc_ci,
    l1_entry_order,
    prefilter_radiomics,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*penalty is deprecated.*", category=UserWarning)

CAND_CLIN_COLS = [SEX_COL, "ALP", "GGT", "Tbil", "Dbil", "INR", "PTZ", "Urea"]


def rank_clinical_additions(df_tr, y_tr, max_m=4):
    X_clin = StandardScaler().fit_transform(df_tr[CAND_CLIN_COLS].values)
    fib4_sc = StandardScaler().fit_transform(df_tr[["FIB4"]].values)
    X_comb = np.column_stack([fib4_sc, X_clin])
    entered = []
    for c_val in np.logspace(-3, 1, 80):
        clf = LogisticRegression(penalty="l1", C=c_val, solver="liblinear",
                                 random_state=SEED, max_iter=500)
        clf.fit(X_comb, y_tr)
        coefs = np.abs(clf.coef_.ravel()[1:])
        nz = np.where(coefs > 1e-5)[0]
        for idx in nz[np.argsort(-coefs[nz])]:
            cname = CAND_CLIN_COLS[idx]
            if cname not in entered:
                entered.append(cname)
                if len(entered) >= max_m:
                    return entered
    for cname in CAND_CLIN_COLS:
        if cname not in entered:
            entered.append(cname)
    return entered[:max_m]


def main():
    parser = argparse.ArgumentParser(description="Fully nested (m, K, C) inner-loop sensitivity analysis.")
    parser.add_argument("--input", default="LBase.txt", help="Tab-delimited input table.")
    parser.add_argument("--outdir", default=".", help="Directory where summary CSV is written.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    y = df[LABEL_COL].to_numpy().astype(int)
    n_samples = len(y)
    df["FIB4"] = (df["Age"] * df["AST"]) / (df["PLT"] * np.sqrt(df["ALT"]))
    radiomic_cols = [c for c in df.columns if c not in ALL_CLINICAL_COLS]

    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=SEED)
    m_grid = [0, 1, 2, 3, 4]
    k_grid = [0, 1, 2, 3, 4, 5]

    oof_max = np.zeros((N_OUTER_REPEATS, n_samples))
    oof_1se = np.zeros((N_OUTER_REPEATS, n_samples))
    chosen_max, chosen_1se = [], []

    for outer_idx, (tr, te) in enumerate(rskf.split(df, y)):
        rep = outer_idx // N_OUTER_SPLITS
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        ytr = y[tr]

        clin_order = rank_clinical_additions(df_tr, ytr, max_m=4)
        infold_rad_pool = prefilter_radiomics(df_tr[radiomic_cols])
        rad_order = l1_entry_order(df_tr[infold_rad_pool], ytr, max_k=5)

        inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=SEED + outer_idx)
        grid_scores = {(m, k, c): [] for m in m_grid for k in k_grid for c in C_GRID}

        all_needed = ["FIB4"] + clin_order + rad_order
        cont_all = [c for c in all_needed if c != SEX_COL]

        for tr_in, val_in in inner_cv.split(df_tr, ytr):
            X_tr_in = df_tr.iloc[tr_in][all_needed].copy()
            X_val_in = df_tr.iloc[val_in][all_needed].copy()
            sc_in = RobustScaler()
            X_tr_in[cont_all] = sc_in.fit_transform(X_tr_in[cont_all])
            X_val_in[cont_all] = sc_in.transform(X_val_in[cont_all])

            for m in m_grid:
                c_feats = ["FIB4"] + clin_order[:m]
                for k in k_grid:
                    feats = c_feats + rad_order[:k]
                    X_tr_mat = X_tr_in[feats].values
                    X_val_mat = X_val_in[feats].values
                    for c_val in C_GRID:
                        clf = LogisticRegression(penalty="l2", C=c_val, solver="liblinear", random_state=SEED)
                        clf.fit(X_tr_mat, ytr[tr_in])
                        p_val = clf.predict_proba(X_val_mat)[:, 1]
                        grid_scores[(m, k, c_val)].append(roc_auc_score(ytr[val_in], p_val))

        mean_scores = {cfg: np.mean(vals) for cfg, vals in grid_scores.items()}
        se_scores = {cfg: np.std(vals, ddof=1) / np.sqrt(N_INNER_SPLITS) for cfg, vals in grid_scores.items()}

        best_max = max(mean_scores.keys(),
                       key=lambda cfg: (round(mean_scores[cfg], 6), -(cfg[0] + cfg[1]), -cfg[2]))
        chosen_max.append(best_max)

        thresh_1se = mean_scores[best_max] - se_scores[best_max]
        eligible_1se = [cfg for cfg, sc_val in mean_scores.items() if sc_val >= thresh_1se]
        best_1se = min(eligible_1se, key=lambda cfg: (cfg[0] + cfg[1], cfg[1], cfg[2], -mean_scores[cfg]))
        chosen_1se.append(best_1se)

        X_tr_out = df_tr[all_needed].copy()
        X_te_out = df_te[all_needed].copy()
        sc_out = RobustScaler()
        X_tr_out[cont_all] = sc_out.fit_transform(X_tr_out[cont_all])
        X_te_out[cont_all] = sc_out.transform(X_te_out[cont_all])

        for cfg, oof_mat in [(best_max, oof_max), (best_1se, oof_1se)]:
            m_sel, k_sel, c_sel = cfg
            feats_sel = ["FIB4"] + clin_order[:m_sel] + rad_order[:k_sel]
            clf = LogisticRegression(penalty="l2", C=c_sel, solver="liblinear", random_state=SEED)
            clf.fit(X_tr_out[feats_sel].values, ytr)
            oof_mat[rep, te] = clf.predict_proba(X_te_out[feats_sel].values)[:, 1]

    out_rows = []
    for rule_name, oof_mat, chosen in [
        ("1SE_Rule", oof_1se, chosen_1se),
        ("Max_Inner_AUC_Rule", oof_max, chosen_max),
    ]:
        p_pool = np.mean(oof_mat, axis=0)
        ci_l, ci_u = bootstrap_auc_ci(y, p_pool, N_BOOT, SEED)
        rep_aucs = [roc_auc_score(y, oof_mat[r]) for r in range(N_OUTER_REPEATS)]
        k_vals = [cfg[1] for cfg in chosen]
        out_rows.append({
            "Rule": rule_name,
            "Pooled_OOF_AUC": roc_auc_score(y, p_pool),
            "CI_95": f"{ci_l:.3f}-{ci_u:.3f}",
            "Repeat_AUC_Mean": np.mean(rep_aucs),
            "Repeat_AUC_SD": np.std(rep_aucs),
            "Brier": brier_score_loss(y, p_pool),
            "Pct_K_le_1": float(np.sum(np.array(k_vals) <= 1)),
            "K_Distribution": str(dict(sorted(Counter(k_vals).items()))),
        })

    res_df = pd.DataFrame(out_rows)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(args.outdir, "inner_loop_sensitivity_summary.csv"), index=False)


if __name__ == "__main__":
    main()
