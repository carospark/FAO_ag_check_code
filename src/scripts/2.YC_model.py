import pandas as pd
import statsmodels.formula.api as smf
from pkg import nakagawa_r2


if __name__ == "__main__":
    
    ### Read in data
    df = pd.read_csv("./data/yield_comparison_df.csv")
    
    ### Transform X covariates for modelling
    coeffs = ["avg_gdp","flag_sum","total_harvarea","cropland_fraction","avg_sm","avg_tmax", "whichlag"]
    df[coeffs].drop(columns="whichlag").agg(['min', 'max'])
    scaled = df[coeffs].drop(columns="whichlag").apply(lambda x: (x - x.mean()) / x.std()).add_suffix("_z")
    df = df.drop(columns=df[coeffs].drop(columns="whichlag")).join(scaled)

    ### Main model
    mod_yields_r2 = smf.mixedlm("r2 ~ avg_gdp_z + flag_sum_z + whichlag + total_harvarea_z + cropland_fraction_z  + avg_sm_z + avg_tmax_z",
                        df, groups="cropname" )
    print(mod_yields_r2.fit().summary())
    nakagawa_r2(mod_yields_r2.fit())

    ### Gnerate model table for Latex
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

    print(summary_df.to_latex(float_format="%.3f"))

