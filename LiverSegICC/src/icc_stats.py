"""Intraclass correlation coefficient (ICC): Shrout & Fleiss (1979), McGraw & Wong (1996).

Y: n subjects x k raters (here a "rater" is a mask: TS, AUTH1, AUTH2).

  ICC(1,1)  one-way random, single measurement
  ICC(2,1)  two-way random, ABSOLUTE AGREEMENT, single measurement   <- primary (Koo & Li 2016)
  ICC(3,1)  two-way mixed, CONSISTENCY, single measurement

95% CIs come from the F distribution (same formulas as pingouin and irr::icc).
selftest.py checks them against Shrout & Fleiss's original worked example.
"""

import math

import numpy as np
from scipy import stats


def _anova(Y):
    n, k = Y.shape
    grand = Y.mean()
    ss_total = ((Y - grand) ** 2).sum()
    ss_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_err) / (n * (k - 1))
    return n, k, msr, msc, mse, msw


def icc_all(Y, alpha = 0.05):
    """All ICC types with 95% CI. Rows with missing values (NaN) are dropped."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    nan = float("nan")
    empty = {t: {"icc": nan, "lo": nan, "hi": nan, "F": nan, "p": nan} for t in ("ICC1", "ICC2", "ICC3")}
    if Y.ndim != 2 or Y.shape[0] < 3 or Y.shape[1] < 2:
        return {"n": int(Y.shape[0]) if Y.ndim == 2 else 0, **empty}
    n, k, msr, msc, mse, msw = _anova(Y)
    out = {"n": n}
    q = 1 - alpha / 2

    # No between-subject variance: ICC is undefined
    if msr <= 0 or not np.isfinite(msr):
        return {"n": n, **empty}

    def safe_div(a, b):
        return a / b if b != 0 else nan

    # ICC(1,1)
    icc1 = safe_div(msr - msw, msr + (k - 1) * msw)
    f1 = safe_div(msr, msw)
    df1, df2_1 = n - 1, n * (k - 1)
    if msw > 0:
        fl = f1 / stats.f.ppf(q, df1, df2_1)
        fu = f1 * stats.f.ppf(q, df2_1, df1)
        lo1, hi1 = (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)
        p1 = stats.f.sf(f1, df1, df2_1)
    else:
        lo1 = hi1 = p1 = nan
    out["ICC1"] = {"icc": icc1, "lo": lo1, "hi": hi1, "F": f1, "p": p1}

    # ICC(3,1)
    df2 = (n - 1) * (k - 1)
    icc3 = safe_div(msr - mse, msr + (k - 1) * mse)
    f3 = safe_div(msr, mse) if mse > 0 else float("inf")
    if mse > 0:
        fl = f3 / stats.f.ppf(q, df1, df2)
        fu = f3 * stats.f.ppf(q, df2, df1)
        lo3, hi3 = (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)
        p3 = stats.f.sf(f3, df1, df2)
    else:
        lo3 = hi3 = 1.0
        p3 = 0.0
    out["ICC3"] = {"icc": icc3, "lo": lo3, "hi": hi3, "F": f3, "p": p3}

    # ICC(2,1)
    icc2 = safe_div(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n)
    if mse > 0 and icc2 < 1:
        a = k * icc2 / (n * (1 - icc2))
        b = 1 + k * icc2 * (n - 1) / (n * (1 - icc2))
        num = (a * msc + b * mse) ** 2
        den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
        v = num / den if den > 0 else nan
        if v and np.isfinite(v) and v > 0:
            fl = stats.f.ppf(q, n - 1, v)
            fu = stats.f.ppf(q, v, n - 1)
            denom_common = k * msc + (k * n - k - n) * mse
            lo2 = n * (msr - fl * mse) / (fl * denom_common + n * msr)
            hi2 = n * (fu * msr - mse) / (denom_common + n * fu * msr)
        else:
            lo2 = hi2 = nan
    else:
        lo2 = hi2 = (1.0 if mse == 0 else nan)
    out["ICC2"] = {"icc": icc2, "lo": lo2, "hi": hi2, "F": f3, "p": p3}
    return out


def koo_li_category(icc):
    """Koo & Li (2016): <0.50 poor, 0.50-0.75 moderate, 0.75-0.90 good, >0.90 excellent."""
    if icc is None or not math.isfinite(icc):
        return "undefined"
    if icc < 0.5:
        return "poor"
    if icc < 0.75:
        return "moderate"
    if icc < 0.9:
        return "good"
    return "excellent"


def bland_altman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 2:
        nan = float("nan")
        return {"bias": nan, "sd": nan, "loa_lo": nan, "loa_hi": nan, "bias_pct": nan, "sd_pct": nan}
    d = a - b
    m = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(m != 0, d / m * 100, np.nan)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {
        "bias": bias, "sd": sd,
        "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
        "bias_pct": float(np.nanmean(pct)), "sd_pct": float(np.nanstd(pct, ddof=1)),
    }
