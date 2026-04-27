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
    title="SM coefficient on satellite yield",
    cbar_label="SM coefficient (dt)",
    cmap="Blues",
    vmin=0, vmax=10,
    filename="satellite_weather_sm",
    hatch_col="pval_sm_dt"
)

plot_map_hatch(
    country_sm[country_sm['model']=="Survey"] ,
    column="coef_sm_dt",
    title="SM coefficient on survey yield",
    cbar_label="SM coefficient (dt)",
    cmap="Blues",
    vmin=0, vmax=10,
    filename="survey_weather_sm",
    hatch_col="pval_sm_dt"
)


# %% [markdown]
# ### Figure 5: Satellite-survey weather-explainability gap vs. GDP per capita

# %%
import matplotlib.ticker as mticker
import pycountry


def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None


fig5 = regs.copy()
gdp = pd.DataFrame(pd.read_csv("./data/wb_gdp_per_cap.csv")[['iso_a3', 'value']])
fig5df = pd.merge(fig5, gdp, on="iso_a3")

wide = fig5df.pivot(index=['iso_a3', 'value'], columns='model', values='adj_r2').reset_index()
wide.columns.name = None
wide = wide.rename(columns={'value': 'gdp_per_cap'})
wide['diff'] = wide['Satellite'] - wide['Survey']
wide['log_gdp'] = np.log(wide['gdp_per_cap'])

centroids = pd.read_csv(
    "https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/dist/countries.csv"
)
centroids['iso_a3'] = centroids['ISO'].apply(iso2_to_iso3)
centroids = centroids.dropna(subset=['iso_a3'])

wide = wide.merge(centroids[['iso_a3', 'latitude']], on='iso_a3', how='left')
wide['tropical'] = wide['latitude'].abs() <= 23.5

mask = np.isfinite(wide['log_gdp']) & np.isfinite(wide['diff'])
wide = wide[mask].copy()

# %%
def _stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '+'
    return ''

def _p_disp(p):
    if p < 0.001: return r"<0.001"
    return f"{p:.3f}".rstrip('0').rstrip('.')

# Overall regression
X = sm.add_constant(wide['log_gdp'])
m_all = sm.OLS(wide['diff'], X).fit()

# By tropical status
m_by_group = {}
for group, label in [(True, 'Tropical'), (False, 'Non-tropical')]:
    sub = wide[wide['tropical'] == group].copy()
    sub = sub[np.isfinite(sub['log_gdp']) & np.isfinite(sub['diff'])]
    X_ = sm.add_constant(sub['log_gdp'])
    m_by_group[label] = (sm.OLS(sub['diff'], X_).fit(), len(sub))

# Joint model with tropical dummy
wide['tropical_int'] = wide['tropical'].astype(int)
X2 = sm.add_constant(wide[['log_gdp', 'tropical_int']])
m_joint = sm.OLS(wide['diff'], X2).fit()

# ---------- LaTeX table: univariate regressions of (Sat-Survey gap) on log GDP ----------
univ_rows = [
    ("Overall",       m_all,                        len(wide)),
    ("Tropical",      m_by_group['Tropical'][0],    m_by_group['Tropical'][1]),
    ("Non-tropical",  m_by_group['Non-tropical'][0], m_by_group['Non-tropical'][1]),
]

latex_lines = [
    r"\begin{table}[!ht]",
    r"\centering",
    r"\begin{tabular}{l r r r r r}",
    r"\toprule",
    r"\multicolumn{6}{c}{Univariate OLS: (Satellite $R^2$ $-$ Survey $R^2$) $\sim$ log GDP per capita} \\",
    r"\midrule",
    r"\textbf{Sample} & \textbf{$\beta_{\log\,GDP}$} & \textbf{SE} & \textbf{CI} & \textbf{p} & \textbf{N} \\",
    r"\midrule",
]
for label, m, n in univ_rows:
    b   = m.params['log_gdp']
    se  = m.bse['log_gdp']
    lo, hi = m.conf_int().loc['log_gdp']
    p   = m.pvalues['log_gdp']
    latex_lines.append(
        f"{label} & {b:.3f} {_stars(p)} & {se:.3f} & {lo:.3f} -- {hi:.3f} & {_p_disp(p)} & {n} \\\\"
    )
latex_lines += [
    r"\bottomrule",
    r"\multicolumn{6}{l}{\footnotesize $^{+}$ p $<0.1$ \quad $^{*}$ p $<0.05$ \quad $^{**}$ p $<0.01$ \quad $^{***}$ p $<0.001$} \\",
    r"\end{tabular}",
    r"\end{table}",
]
print("\n".join(latex_lines))

# ---------- LaTeX table: joint model (log GDP + tropical dummy) ----------
joint_rows = [
    ("Intercept",       'const'),
    ("log GDP/capita",  'log_gdp'),
    ("Tropical (1/0)",  'tropical_int'),
]

latex_lines = [
    r"",
    r"\begin{table}[!ht]",
    r"\centering",
    r"\begin{tabular}{l r r r r}",
    r"\toprule",
    r"\multicolumn{5}{c}{Joint OLS: (Satellite $R^2$ $-$ Survey $R^2$) $\sim$ log GDP $+$ Tropical} \\",
    r"\midrule",
    r"\textbf{Predictor} & \textbf{Estimate} & \textbf{SE} & \textbf{CI} & \textbf{p} \\",
    r"\midrule",
]
for label, term in joint_rows:
    b  = m_joint.params[term]
    se = m_joint.bse[term]
    lo, hi = m_joint.conf_int().loc[term]
    p  = m_joint.pvalues[term]
    latex_lines.append(
        f"{label} & {b:.3f} {_stars(p)} & {se:.3f} & {lo:.3f} -- {hi:.3f} & {_p_disp(p)} \\\\"
    )
latex_lines += [
    r"\midrule",
    fr"$R^2$ / Adj $R^2$ & {m_joint.rsquared:.3f} / {m_joint.rsquared_adj:.3f} & & & \\",
    fr"F-statistic ({int(m_joint.df_model)}, {int(m_joint.df_resid)}) & {m_joint.fvalue:.2f} & & & p = {_p_disp(m_joint.f_pvalue)} \\",
    fr"Observations & {int(m_joint.nobs)} & & & \\",
    r"\bottomrule",
    r"\multicolumn{5}{l}{\footnotesize $^{+}$ p $<0.1$ \quad $^{*}$ p $<0.05$ \quad $^{**}$ p $<0.01$ \quad $^{***}$ p $<0.001$} \\",
    r"\end{tabular}",
    r"\end{table}",
]
print("\n".join(latex_lines))

# %%
fig, ax = plt.subplots(figsize=(7, 5))

for trop, label, color in [(True, 'Tropical', 'firebrick'), (False, 'Non-tropical', 'gray')]:
    sub = wide[wide['tropical'] == trop]
    ax.scatter(sub['gdp_per_cap'], sub['diff'], alpha=0.5, s=25,
               color=color, linewidths=0, label=label)

x_range = np.linspace(wide['log_gdp'].min(), wide['log_gdp'].max(), 300)
for trop, color in [(True, 'firebrick'), (False, 'gray')]:
    sub = wide[wide['tropical'] == trop]
    X_ = sm.add_constant(sub['log_gdp'])
    m = sm.OLS(sub['diff'], X_).fit()
    y_hat = m.params['const'] + m.params['log_gdp'] * x_range
    ax.plot(np.exp(x_range), y_hat, color=color, linewidth=1.8)

y_hat_all = m_all.params['const'] + m_all.params['log_gdp'] * x_range
ax.plot(np.exp(x_range), y_hat_all, color='black', linewidth=1.8,
        linestyle='--', label='All countries')

ax.set_xscale('log')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax.set_xlabel('GDP per capita (log scale, USD)', fontsize=12)
ax.set_ylabel('Satellite R² − Survey R²\n(avg. across crops)', fontsize=12)
ax.set_title('Satellite–survey weather explainability gap\nvs. GDP per capita', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('./plots/fig5_gdp_scatter.pdf')
plt.show()

