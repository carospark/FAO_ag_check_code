import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pkg import nakagawa_r2
import os
import re


def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '+'
    return ''

def p_display(p):
    if p < 0.001: return r"<0.001"
    return f"{p:.3f}".rstrip('0').rstrip('.')

def ci_fmt(lo, hi, ndigits=2):
    return f"{lo:.{ndigits}f} – {hi:.{ndigits}f}"

def clean_whichlag(term):
    m = re.search(r"C\(whichlag.*?\)\[T\.(.+?)\]$", term)
    if m:
        return f"Offset ({m.group(1)})"
    return term

NICE = {
    "Intercept": "Intercept",
    "avg_gdp_z": "Avg GDP/capita",
    "flag_pct_z": "Survey flag pct",
    "cropland_fraction_z": "Cropland fraction",
    "total_harvarea_z": "Total harvested area",
    "whichlag": "Offset (year ± 1)"
}


def fit_and_emit_table(df, dep_var, header_math):
    """Fit the mixed-effects model with `dep_var` as outcome and print a LaTeX table."""
    formula = f"{dep_var} ~ avg_gdp_z + flag_pct_z + whichlag + total_harvarea_z + cropland_fraction_z"
    res = smf.mixedlm(formula, df, groups="cropname").fit()

    params = res.fe_params
    bse    = res.bse_fe
    conf   = res.conf_int().loc[params.index]
    conf.columns = ["lo", "hi"]
    tvals  = params / bse
    pvals  = res.pvalues.loc[params.index] if hasattr(res, "pvalues") else np.nan

    fe = (
        pd.DataFrame({
            "term": params.index,
            "est": params.values,
            "se": bse.values,
            "t": tvals.values,
            "p": pvals.values
        })
        .merge(conf, left_on="term", right_index=True, how="left")
    )

    is_whichlag = fe["term"].str.contains(r"^C\(whichlag")
    fe_intercept = fe[fe["term"] == "Intercept"]
    fe_main      = fe[(fe["term"] != "Intercept") & (~is_whichlag)]
    fe_lags      = fe[is_whichlag]
    ordered = pd.concat([fe_intercept, fe_main, fe_lags], ignore_index=True)

    rows = []
    for _, r in ordered.iterrows():
        term = r["term"]
        label = NICE.get(term, term)
        if term.startswith("C(whichlag"):
            label = clean_whichlag(term)
        stars = p_to_stars(r["p"])
        est_s = f"{r['est']:.2f} {stars}".rstrip()
        ci_s  = ci_fmt(r["lo"], r["hi"], ndigits=2)
        p_s   = p_display(r["p"])
        rows.append((label, est_s, ci_s, p_s))

    sigma2 = float(res.scale)
    try:
        tau00 = float(np.asarray(res.cov_re)[0, 0])
    except Exception:
        tau00 = np.nan
    icc = tau00 / (tau00 + sigma2) if np.isfinite(tau00) else np.nan
    n_groups = int(df["cropname"].nunique())
    n_obs    = int(len(res.model.endog))
    r2_marg, r2_cond = nakagawa_r2(res)

    def fmt_or_blank(x):
        return "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    latex_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\begin{tabular}{l l l l}",
        r"\toprule",
        fr"\multicolumn{{4}}{{c}}{{{header_math}}} \\",
        r"\midrule",
        r"\textbf{Predictors} & \textbf{Estimates} & \textbf{CI} & \textbf{p} \\",
        r"\midrule",
    ]
    for label, est_s, ci_s, p_s in rows:
        latex_lines.append(f"{label} & {est_s} & {ci_s} & {p_s} \\\\")
    latex_lines += [
        r"\midrule",
        r"\textbf{Random Effects} & & & \\",
        fr"$\sigma^2$ & {fmt_or_blank(sigma2)} & & \\",
        fr"$\tau_{00}$ cropname & {fmt_or_blank(tau00)} & & \\",
        fr"ICC & {fmt_or_blank(icc)} & & \\",
        fr"N cropname & {n_groups} & & \\",
        fr"Observations & {n_obs} & & \\",
        fr"Marginal R2 / Conditional R2 & {fmt_or_blank(r2_marg)} \, / \, {fmt_or_blank(r2_cond)} & & \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize $^{+}$ p $<0.1$ \quad $^{*}$ p $<0.05$ \quad $^{**}$ p $<0.01$ \quad $^{***}$ p $<0.001$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    print("\n".join(latex_lines))


if __name__ == "__main__":

    ### Read in data
    df = pd.read_csv("./data/yield_comparison_df.csv")

    ### Transform X covariates for modelling
    coeffs = ["avg_gdp","flag_pct","total_harvarea","cropland_fraction","avg_sm","avg_tmax", "whichlag"]
    scaled = df[coeffs].drop(columns="whichlag").apply(lambda x: (x - x.mean()) / x.std()).add_suffix("_z")
    df["whichlag"] = df["whichlag"].isin(["yield_lag", "yield_lead"]).astype(int)
    df = df.drop(columns=df[coeffs].drop(columns="whichlag")).join(scaled)

    ### Main model: rho^2 (Table 2 in manuscript)
    fit_and_emit_table(df, dep_var="r2",
                       header_math=r"$\rho^2_{VS}$: correlation squared of survey \& satellite")

    ### Supplementary model: rho (unsquared) -- referenced in manuscript as
    ### "results are consistent when analyzing rho_VS rather than rho^2_VS"
    print("\n\n")
    fit_and_emit_table(df, dep_var="rho",
                       header_math=r"$\rho_{VS}$: correlation of survey \& satellite (supplementary)")
