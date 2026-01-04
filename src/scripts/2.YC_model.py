import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pkg import nakagawa_r2
import os
import re


if __name__ == "__main__":
    
    ### Read in data
    df = pd.read_csv("./data/yield_comparison_df.csv")

    ### Transform X covariates for modelling
    coeffs = ["avg_gdp","flag_sum","total_harvarea","cropland_fraction","avg_sm","avg_tmax", "whichlag"]
    #df[coeffs].drop(columns="whichlag").agg(['min', 'max'])
    scaled = df[coeffs].drop(columns="whichlag").apply(lambda x: (x - x.mean()) / x.std()).add_suffix("_z")
    df["whichlag"] = df["whichlag"].isin(["yield_lag", "yield_lead"]).astype(int)
    df = df.drop(columns=df[coeffs].drop(columns="whichlag")).join(scaled)

    ### Main model

    mod_yields_r2 = smf.mixedlm("r2 ~ avg_gdp_z + flag_sum_z + whichlag + total_harvarea_z + cropland_fraction_z ",
                    df, groups="cropname" )

    res = mod_yields_r2.fit()
    params = res.params
    conf = res.conf_int()
    summary_df = (
        pd.DataFrame({
            "Coef.": params,
            "Std.Err.": res.bse,
            "t": res.tvalues,
            "P>|t|": res.pvalues,
            "CI Lower": conf[0],
            "CI Upper": conf[1]
        }))


    ######### code below is just to make a pretty latex table out of the raw output
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
        # turns C(whichlag, Treatment(reference='yield_log_dt'))[T.something]
        # into a readable label like "Offset (something)"
        m = re.search(r"C\(whichlag.*?\)\[T\.(.+?)\]$", term)
        if m:
            return f"Offset ({m.group(1)})"
        return term

    # Nice labels for fixed effects (add/adjust as you wish)
    NICE = {
        "Intercept": "Intercept",
        "avg_gdp_z": "Avg GDP/capita",
        "flag_sum_z": "Survey flag pct",
        "cropland_fraction_z": "Cropland fraction",
        "total_harvarea_z": "Total harvested area",
        "whichlag": "Offset (year ± 1)"
    }

    # ---------- gather model pieces ----------
    res = mod_yields_r2.fit()
    params = res.fe_params          # fixed effects only
    bse    = res.bse_fe             # std. errors for fixed effects
    conf   = res.conf_int().loc[params.index]  # keep only FE rows
    conf.columns = ["lo", "hi"]

    # p-values and t-values for fixed effects
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


    # build display rows
    rows = []

    # order: Intercept first, then everything except whichlag contrasts, then whichlag contrasts
    is_whichlag = fe["term"].str.contains(r"^C\(whichlag")
    fe_intercept = fe[fe["term"] == "Intercept"]
    fe_main      = fe[(fe["term"] != "Intercept") & (~is_whichlag)]
    fe_lags      = fe[is_whichlag]

    ordered = pd.concat([fe_intercept, fe_main, fe_lags], ignore_index=True)

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

    # ---------- random-effects & fit stats ----------
    # residual variance (sigma^2)
    sigma2 = float(res.scale)

    # random intercept variance tau00 (first element of cov_re)
    try:
        tau00 = float(np.asarray(res.cov_re)[0,0])
    except Exception:
        tau00 = np.nan

    # ICC
    icc = tau00 / (tau00 + sigma2) if np.isfinite(tau00) else np.nan

    # counts
    n_groups = int(df["cropname"].nunique())
    n_obs    = int(len(res.model.endog))

    # Nakagawa R2 (if you have a function; else set to np.nan)
    r2_marg, r2_cond = nakagawa_r2(res)        

    def fmt_or_blank(x):
        return "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    # ---------- emit LaTeX ----------
    latex_lines = []
    latex_lines += [
    r"\begin{table}[!ht]",
    r"\centering",
    r"\begin{tabular}{l l l l}",
    r"\toprule",
    r"\multicolumn{4}{c}{$\rho^2_{VS}$: correlation squared of survey \& satellite} \\",
    r"\midrule",
    r"\textbf{Predictors} & \textbf{Estimates} & \textbf{CI} & \textbf{p} \\",
    r"\midrule"
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
    r"\end{table}"
    ]

    print("\n".join(latex_lines))

