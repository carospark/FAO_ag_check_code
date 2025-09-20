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


__all__ = ["pearson_corr", "regress"]