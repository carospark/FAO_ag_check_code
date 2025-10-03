from pkg import first_clean, fetch_ag_covariates, flexible_merge, detrend_group, pearson_corr
from functools import reduce
import pandas as pd
import os 

os.chdir("/Users/caropark/FAO_ag_check_code")

### run only if you want to regenerate satellite-based covariates. takes a long time!
def agricultural_covariates():
    path = "SET PATH HERE"
    allcountries = first_clean(path)

    combined= pd.DataFrame()

    for country in allcountries:
        print(country)
        ag_covs = fetch_ag_covariates(country)
        combined = combined.append(ag_covs)
        print("\n")
        
    key = pd.read_pickle("./data/calendar_fao_cropkey.pkl")[["crop", "cropname"]]
    combined = combined.merge(key)
    ag = combined.reset_index(drop=True)
    #ag.to_csv("./data/ag_vars.csv")


if __name__ == "__main__":
    ### To generate X independent variables:
    ######### read in covariate data
    ag = pd.read_csv("./data/ag_vars.csv")[['total_harvarea', 'cropland_fraction', 'cropname', 'iso_a3']]

    gdp = pd.read_csv("./data/wb_gdp_per_cap.csv").rename({'value': 'gdp'}, axis=1)
    gdp = gdp.groupby(['iso_a3']).agg(avg_gdp = ('gdp', 'mean')).reset_index()

    flag = pd.read_csv("./data/faostat_all_flags.csv").rename({"Year": 'year'}, axis=1)
    flag = flag.groupby(["iso_a3", 'cropname'])[['year']].agg(flag_sum = ('year', 'count')).reset_index()

    clim = (pd.read_csv("./data/dt_clim_vars.csv").groupby(['iso_a3', 'cropname'])[['sm', 'tmax']]
            .agg(avg_sm = ('sm', 'mean'), avg_tmax= ('tmax', 'mean')).reset_index())
    
    ### To generate Y dependent variable:
    ######### read in yield data, detrend, and then first stage regression to get R2 
    yields = pd.read_csv("./data/yields.csv")[["cropname", "year", "yield_og", "csif_og", "whichlag", "country", "iso_a3"]]    
    yields = detrend_group(yields, 'yield_og', 'yield_log_dt', log_transform=True)
    yields = detrend_group(yields, 'csif_og', 'csif_log_dt', log_transform=True)
    yield_r2 = yields.groupby(['iso_a3', 'cropname', 'whichlag']).apply(lambda group: pd.Series({"r2": pearson_corr(group, x="csif_log_dt",y="yield_log_dt")**2})).reset_index()
    yield_r2 = yield_r2[yield_r2['r2']!=1]

    ### To generate final DF for modelling:
    ######### merge X covariates with R2 dependent variable
    dfs = [yield_r2, gdp, flag, ag, clim]

    merged = reduce(flexible_merge, dfs).dropna(subset=["r2","avg_sm"]).drop_duplicates().reset_index(drop=True)
    merged["flag_sum"] = merged["flag_sum"].fillna(0)

    coeffs = ["avg_gdp","flag_sum","total_harvarea","cropland_fraction","avg_sm","avg_tmax", "whichlag"]
    merged = merged.dropna(subset=coeffs.copy().extend(["r2", "cropname"])).reset_index(drop=True)

    merged.to_csv("./data/yield_comparison_df.csv")