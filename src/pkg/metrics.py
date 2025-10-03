import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm    

def pearson_corr(df: pd.DataFrame, x: str, y: str) -> float:
    sub = df[[x, y]].dropna()
    if len(sub) < 2:
        return np.nan
    return sub.corr().iloc[0, 1] ## off-diagonal = correlation


def nakagawa_r2(mod_fit):       # for mixed LM
    var_f = np.var(mod_fit.fittedvalues, ddof=1)   # variance explained by fixed effects, ddof = divides by n-1 (sample variance, unbiased)
    var_r = mod_fit.cov_re.iloc[0, 0]              # random intercept variance
    var_e = mod_fit.scale                          # residual variance (error term)

    marginal_r2   = var_f / (var_f + var_r + var_e)
    conditional_r2 = (var_f + var_r) / (var_f + var_r + var_e)

    print("Marginal R² =", marginal_r2)
    print("Conditional R² =", conditional_r2)


def detrend_group(df, column, newname, time_column="year", log_transform=False):
    df = df.copy()
    df[time_column] = pd.to_numeric(df[time_column], errors="coerce")
    
    def detrend(sub_df):
        sub_df_nonan = sub_df.dropna(subset=[column, time_column])
        
        if sub_df_nonan[time_column].nunique() < 2:
            vals = sub_df_nonan[column]
            if log_transform:
                vals = np.log(vals.replace(0, np.nan))  # handle zeros safely
            sub_df[newname] = vals - vals.mean()
            return sub_df
        
        X = sm.add_constant(sub_df_nonan[time_column])
        y = sub_df_nonan[column]
        if log_transform:
            y = np.log(y.replace(0, np.nan))  # avoid log(0) crash
        
        model = sm.OLS(y, X).fit()
        sub_df.loc[sub_df_nonan.index, newname] = model.resid
        
        return sub_df
    
    df = df.groupby(["country", "cropname"], group_keys=False).apply(detrend)
    return df


# def regress(data, yvar, xvars):
#     Y = data[yvar]
#     X = data[xvars]
#     X['intercept'] = 1.
#     result = sm.OLS(Y, X).fit()
#     residual_var = result.mse_resid
#     return pd.Series([result.params[0], result.rsquared, residual_var])



__all__ = ["pearson_corr", "nakagawa_r2", "detrend_group"]