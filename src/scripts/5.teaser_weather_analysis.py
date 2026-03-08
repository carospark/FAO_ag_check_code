# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
from pkg import detrend_group, plot_map, plot_map_hatch
from scipy.stats import norm


# %%
def safe_regress(data, formula=None, yvar=None, xvars=None, ci=None):
    if formula:
        yvar = formula.split("~")[0].strip()
        xvars = formula.split("~")[1].strip().split("+")
        xvars = [x.strip() for x in xvars]


    if formula is None and yvar is None or xvars is None:
        raise ValueError("Provide either formula or yvar + xvars")

    n_obs = len(data.dropna(subset=[yvar] + xvars))
    n_preds = len(xvars) + 1  # +1 for intercept

    if n_obs <= n_preds:
        return pd.DataFrame([{"r2": None, "adj_r2": None}])

    try:
        if formula:
            model = smf.ols(formula=formula, data=data).fit()
        else:
            formula = f"{yvar} ~ {' + '.join(xvars)}"
            model = smf.ols(formula=formula, data=data).fit()

        out = {"r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "ftest_pval": model.f_pvalue}
        
        out.update({f"coef_{k}": v for k, v in model.params.items()})
        out.update({f"pval_{k}": v for k, v in model.pvalues.items()})
        
        if ci is not None:
            ci = model.conf_int()
            for k in model.params.index:
                out[f"cilow_{k}"] = ci.loc[k, 0]
                out[f"cihigh_{k}"] = ci.loc[k, 1]

        return pd.DataFrame([out])

    except Exception as e:
        return pd.DataFrame([{"r2": None, "adj_r2": None, "error": str(e)}])

def pval_is_sig(data, column, threshold = 0.05):
    data[f'{column}_pass'] = np.where(data[column] <0.05, True, False )
    return data

def make_pretty_tab_multi(df, cols):
    out = {}
    for col in cols:
        vals = {'low': df.loc[df['quantile']==0.25, col].item(),
            'med': df.loc[df['quantile']==0.5,  col].item(),
            'high': df.loc[df['quantile']==0.75, col].item()}
        out[col] = f"{vals['med']} ({vals['low']}, {vals['high']})"
    return pd.Series(out)


# %%
yields = pd.read_csv("./data/yields.csv")
yields= yields.loc[:, ~yields.columns.str.contains("_lag|_lead|fao_idx|gridcells", regex=True)]

sm_tmax = pd.read_csv("./data/sm_tmax.csv")
yields_clim = yields.merge(sm_tmax, how="left", on = ["year", "cropname", "country"])

yields_clim = detrend_group(yields_clim, "sm_og", "sm_dt")
yields_clim = detrend_group(yields_clim, "tmax_og", "tmax_dt")

yields_clim = yields_clim[yields_clim.notna()]

counts = yields_clim[["country", "cropname"]].value_counts()
counts_idx = counts[counts>10].index
yields_clim10 = yields_clim.set_index(['country', 'cropname'])
yields_clim10 =  yields_clim10.loc[counts_idx].reset_index()

# %%
res_surv = yields_clim10.groupby(['cropname', 'country']).apply(
    lambda group: safe_regress(group, formula= "yield_log_dt ~ sm_dt + tmax_dt")
    ).reset_index(level=[0,1])
res_surv = res_surv.iloc[:, ~res_surv.columns.str.contains("Intercept", regex=True)]
res_surv['model']="Survey"

res_sat = yields_clim10.groupby(['cropname', 'country']).apply(
    lambda group: safe_regress(group, formula= "csif_log_dt ~ sm_dt + tmax_dt")
    ).reset_index(level=[0,1])
res_sat = res_sat.iloc[:, ~res_sat.columns.str.contains("Intercept", regex=True)]
res_sat['model'] = "Satellite"

res_comb= pd.concat([res_sat, res_surv]).reset_index(drop=True)

[pval_is_sig(res_comb, col) for col in ['pval_sm_dt', 'pval_tmax_dt', 'ftest_pval']]
res_comb= res_comb.dropna(how="any")


# %%
cols = ["adj_r2", "r2", "coef_sm_dt", "coef_tmax_dt"]
nice_tab1_allcrops = res_comb.groupby('model')[['r2', 'adj_r2', 'coef_sm_dt', 'coef_tmax_dt']
                                               ].quantile([0.25, 0.5, 0.75]).round(2).reset_index(
                                               ).rename({'level_1': 'quantile'}, axis=1).groupby('model').apply(lambda x: make_pretty_tab_multi(x, cols))
nice_tab2_allcrops = res_comb.groupby('model')[['pval_sm_dt_pass', 'pval_tmax_dt_pass', 'ftest_pval_pass']].agg(lambda x: str(int(round((sum(x)/ len(x)*100),0)))+"%")


nice_tab_allcrops = pd.merge(nice_tab1_allcrops, nice_tab2_allcrops, right_on = "model", left_on="model")
nice_tab_allcrops['cropname'] = "All crops"
nice_tab_allcrops = nice_tab_allcrops.reset_index().set_index(['cropname', 'model'])


# %%
wanted = res_comb[res_comb['cropname'].isin(["Maize", "Sorghum", "Wheat", "Potatoes", "Cassava"]) ]

nice_tab1 = wanted.groupby(['cropname', 'model'])[['r2', 'adj_r2', 'coef_sm_dt', 'coef_tmax_dt']
                                                  ].quantile([0.25, 0.5, 0.75]).round(2).reset_index(
                                                  ).rename({'level_2': 'quantile'}, axis=1
                                                  ).groupby(['cropname', 'model']).apply(lambda x: make_pretty_tab_multi(x, cols))

nice_tab2= wanted.groupby(['cropname', 'model'])[['pval_sm_dt_pass', 'pval_tmax_dt_pass', 'ftest_pval_pass']].agg(lambda x: str(int(round((sum(x)/ len(x)*100),0)))+ "%")
nice_table =  pd.merge(nice_tab1, nice_tab2, left_on=(['cropname', "model"]), right_on=(['cropname', 'model']))
nice_table = pd.concat([nice_tab_allcrops, nice_table])

# %%
order = ["All crops", "Maize", "Sorghum", "Wheat", "Cassava", "Potatoes"]
nice_table = nice_table.iloc[: , [1,0, 6, 2,4]].loc[order,:]
nice_table.columns=['R2', "Adj R2", "F-test pass", "SM coefficient", "SM p.v. pass"]


# %%
table = nice_table.copy()
cols_to_clean = ["R2", "Adj R2"]
table[cols_to_clean] = table[cols_to_clean].replace(r"\s*\([^)]*\)", "", regex=True)

print(table.to_latex())

# %%
full_table = nice_table.replace(r"(\d+\.\d+)\s\(([^)]+)\)", r"\\makecell{\1 \\\\ (\2)}", regex=True)
full_table = full_table.replace("%", r"\\%", regex=True)

print(full_table.to_latex(escape=False))

# %% [markdown]
# ### country map

# %%
country_key = pd.read_csv("./data/country_key.csv")

regs = res_comb[['cropname', 'country', 'adj_r2', 'coef_sm_dt', 'model']
                                                    ].merge(country_key[["iso_a3", "country"]], how="left", on="country")
wb_class = pd.read_csv("./data/wb_classification.csv")[['iso_a3', "class"]]
regs= regs.merge(wb_class, how="left", on="iso_a3")
regs = regs.groupby(['iso_a3', 'model']).mean('adj_r2').reset_index()
#regs.to_csv("./data/fig4_adjr2.csv")

# %%

plot_map(regs[regs['model']=="Survey"], column="adj_r2", 
         title="Survey yield variability explainable by SM and TMAX", 
         cbar_label="Average adjusted R2", cmap="plasma_r", vmin=0, vmax=1,
         filename="adj_r2_census_allcrops")

# %%

plot_map(regs[regs['model']=="Satellite"], column="adj_r2", 
         title="Satellite yield variability explainable by SM and TMAX", 
         cbar_label="Average adjusted R2", cmap="plasma_r", vmin=0, vmax=1,
         filename="adj_r2_csif_allcrops")


## SM maps

regs_sm = res_comb[['cropname', 'country', 'coef_sm_dt', 'pval_sm_dt', 'model']
                                                    ].merge(country_key[["iso_a3", "country"]], how="left", on="country")
wb_class = pd.read_csv("./data/wb_classification.csv")[['iso_a3', "class"]]
regs_sm= regs_sm.merge(wb_class, how="left", on="iso_a3")

df = regs_sm.copy() 

# recover t-stat
df['t'] = norm.ppf(1 - df['pval_sm_dt']/2)
# recover standard error
df['se'] = np.abs(df['coef_sm_dt'] / df['t'])
# inverse variance weight
df['w'] = 1 / df['se']**2

country_coef = (
    df.groupby(['iso_a3', 'model'])
      .apply(lambda g: np.sum(g.coef_sm_dt * g.w) / np.sum(g.w))
      .rename('coef_sm_dt')
)

def combine_p(g):
    z = np.sign(g.coef_sm_dt) * norm.ppf(1 - g.pval_sm_dt/2)
    zc = np.sum(z * np.sqrt(g.w)) / np.sqrt(np.sum(g.w))
    return 2*(1 - norm.cdf(abs(zc)))

country_p = df.groupby(['iso_a3', 'model']).apply(combine_p).rename('pval_sm_dt')

country_sm = pd.concat([country_coef, country_p], axis=1).reset_index()


plot_map_hatch(
    country_sm[country_sm['model']=="Satellite"] ,
    column="coef_sm_dt",
    title="SM coefficient on survey yield",
    cbar_label="SM coefficient (dt)",
    cmap="plasma_r",
    vmin=0, vmax=10,
    filename="satellite_weather_sm",
    hatch_col="pval_sm_dt"
)

plot_map_hatch(
    country_sm[country_sm['model']=="Survey"] ,
    column="coef_sm_dt",
    title="SM coefficient on survey yield",
    cbar_label="SM coefficient (dt)",
    cmap="plasma_r",
    vmin=0, vmax=10,
    filename="survey_weather_sm",
    hatch_col="pval_sm_dt"
)

