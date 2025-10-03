import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pkg import pearson_corr, plot_twin_lines
plt.rcParams.update({'font.size': 22})

if __name__ == "__main__":


    ################################# USA plots: best and worst survey-satellite relationships

    ### read in data and sort by correlation:
    og = pd.read_csv("./data/yields.csv")
    usa = og[og['country']=="United States of America"]
    corrs = (usa.groupby("cropname", group_keys=False)
            .apply(lambda d: pearson_corr(d, "csif_log_dt", "yield_log_dt"))
            .sort_values(ascending=False))
    sorted_crops = corrs.index.tolist()

    ### plot best survey-satellite relationships
    colors1 = ["green", "orange", "#98ff98", "#d0aaf3", "pink"]
    best = usa[usa['cropname'].isin(sorted_crops[:5])]
    plot_twin_lines(best, col="cropname", col_order = sorted_crops[:5], col_wrap=5, 
                    grid_kwargs={"hue":"cropname", 'palette': colors1}, plot_kwargs= None, filename="usa_ts_best")


    colors2 = ["maroon", "#cbbeb5", "#aa629c", "#9bd18c", "#2acaea"]
    worst = usa[usa['cropname'].isin(sorted_crops[-5:])]
    plot_twin_lines(worst, col="cropname", col_order = sorted_crops[-5:], col_wrap=5, 
                    grid_kwargs={"hue":"cropname", 'palette': colors2}, plot_kwargs= None, filename= "usa_ts_worst")
    

    ################################# global plots: FAOSTAT maize flags

    ### filter for maize time series with complete data
    maize_18 = (og.query("cropname == 'Maize' and `yield_log_dt` == `yield_log_dt`") 
        .assign(counts=lambda d: d.groupby('iso_a3')['iso_a3'].transform('size'))
        .query("counts > 17").copy().rename(columns={"year": 'Year'}))

    ### sort by correlation
    corrdf = maize_18.groupby(['country', 'iso_a3']).apply(lambda d:pd.Series(stats.pearsonr(d['csif_log_dt'], d['yield_log_dt']),
                                                                            index=["corr", "pval"])).reset_index()
    corrdf = corrdf.assign(pval= lambda d: round(d['pval'], 4), **{"abs(corr)": lambda d: abs(d['corr'])})
    corr_idx = corrdf.sort_values("corr").reset_index()[['country', 'iso_a3']]

    flags = (pd.read_csv("./data/faostat_all_flags.csv")[lambda df: df['Flag Description'] == "Estimated value"]
        .query("cropname == 'Maize' "))[['iso_a3', 'country', 'Year', 'cropname']].rename({"Year":"year"}, axis=1).drop_duplicates()
    flags['flag'] = pd.to_numeric(1)


    ### highest survey-satellite correlations for maize:
    df = maize_18[maize_18['iso_a3'].isin(corr_idx.iloc[-5:,1])]
    flags_best = flags[flags['iso_a3'].isin(corr_idx.iloc[-5:,1])]
    flags_best.empty ## check to make sure there aren't flags

    plot_twin_lines(df, col="country", col_wrap = 5, col_order = corr_idx.iloc[-5:,0], grid_kwargs=None, plot_kwargs={'color': "red"}, filename="global_maize_best_ts.pdf")

    ### lowest survey-satellite correlations for maize:
    df = maize_18[maize_18['country'].isin(corr_idx.iloc[:5,0])]
    flags_worst = flags[flags["iso_a3"].isin(corr_idx.iloc[:5,1])]
    flags_worst.groupby('iso_a3')[['year']].agg([min, max, len])
    fl = flags_worst[["country", 'year']].values.tolist()

    plot_twin_lines(df, col="country", col_wrap = 5, col_order = corr_idx.iloc[:5,0], grid_kwargs=None, plot_kwargs={'color': "red"}, 
                    filename="global_maize_worst_ts.pdf", fl=fl)
