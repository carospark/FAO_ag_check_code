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


def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    residual_var = result.mse_resid
    return pd.Series([result.params[0], result.rsquared, residual_var])


def nakagawa_r2(mod_fit):       # for mixed LM
    var_f = np.var(mod_fit.fittedvalues, ddof=1)   # variance explained by fixed effects, ddof = divides by n-1 (sample variance, unbiased)
    var_r = mod_fit.cov_re.iloc[0, 0]              # random intercept variance
    var_e = mod_fit.scale                          # residual variance (error term)

    marginal_r2   = var_f / (var_f + var_r + var_e)
    conditional_r2 = (var_f + var_r) / (var_f + var_r + var_e)

    print("Marginal R² =", marginal_r2)
    print("Conditional R² =", conditional_r2)



__all__ = ["pearson_corr", "regress", "nakagawa_r2"]