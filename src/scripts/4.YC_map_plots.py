import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pkg import plot_map
plt.rcParams.update({'font.size': 22})

if __name__ == "__main__":

    ################################# processing for plots

    ### sat-surv correlation squared (first-stage R2)
    yield_comp = pd.read_csv("./data/yield_comparison_df.csv")
    r2_best = yield_comp.groupby('iso_a3')['r2'].max().reset_index()
    r2_avg = yield_comp.groupby('iso_a3')['r2'].mean().reset_index()
    r2_maize = yield_comp.query('cropname == "Maize" ')

    ## gdp data
    gdp_df = pd.read_csv("./data/wb_gdp_per_cap.csv")

    ## lead/lag analysis
    leadlag = (pd.DataFrame(yield_comp[["cropname", "iso_a3", "whichlag"]].drop_duplicates().reset_index(drop=True)
                    .groupby('iso_a3')['whichlag'].value_counts(normalize=True).round(2))
                    .rename(columns={'whichlag':'pct'}).reset_index().drop_duplicates())

    leadlag_majority = leadlag.groupby('iso_a3')['pct'].idxmax()
    leadlag_cat = leadlag.loc[leadlag_majority.dropna()]
    leadlag_pct = leadlag.copy()
    leadlag_pct['whichlag'] = leadlag_pct['whichlag'].replace({'yield_lag': 'lead/lag', 'yield_lead': 'lead/lag'})
    leadlag_pct = leadlag_pct.groupby(['iso_a3', 'whichlag']).sum('pct').reset_index()
    leadlag_pct = leadlag_pct[leadlag_pct['whichlag']=="lead/lag"]
    leadlag_pct['pct'] = 1-leadlag_pct['pct']
    
    
    ## fao flag analysis
#     fao = (pd.read_csv("./data/faostat_all_flags.csv")[lambda df: df['Flag Description'] == "Estimated value"]
#         .query("cropname == 'Maize' "))
#     fao['flag'] = pd.to_numeric(1)
#     fao = fao.groupby('iso_a3').sum('flag')['flag'].reset_index()

     ## fao flag analysis
     

################################# plots
    
    ### sat-surv correlation squared (first-stage R2)
    plot_map(r2_best, column="r2", 
         title="Strongest Correspondence b/w FAO-reported yields & CSIF proxy for all crops", 
         cbar_label="Correspondence Index", cmap="plasma_r", 
         filename="best_corr_map")
    plot_map(r2_avg, column="r2", 
            title="Average Correspondence b/w FAO-reported yields & CSIF proxy for all crops", 
            cbar_label="Averaged Correspondence Index", cmap="plasma_r", 
            filename="avg_corr_map")
    plot_map(r2_maize, column="r2", 
            title="Strongest Correspondence b/w FAO-reported yields & CSIF proxy for all crops", 
            cbar_label="Averaged Correspondence Index", cmap="plasma_r", 
            filename="maize_corr_map")

    ### GDP map
    bounds = [0, 1000, 2000, 5000, 10000, 20000, 50000]
    cmap = plt.cm.YlGnBu
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    plot_map(gdp_df, column="value", 
            title="Average GDP/capita over study period", 
            cmap="YlGnBu", cbar_label="Average GDP/capita", norm=norm,
            filename="avg_gdp_map")

    ### lag map
    colors = ["#ffcba4", "#ca8af2", "#84d7bb"]
    newcmp = ListedColormap(colors)
    plot_map(leadlag_cat, column="whichlag", 
         title="Majority lead/lag relationship between CSIF proxy & FAO-reported yields", 
         cmap=newcmp, cbar_label=None,
         filename="leadlag_map")

    ### lag map (pct)
    colors = ["lightyellow", "#f3f380", "#6bc504", "green", "darkgreen"]
    new_cmap = mcolors.LinearSegmentedColormap.from_list("YellowToGreen", colors)
    plot_map(leadlag_pct, column="pct", 
         title="Harvest year consistency b/w FAO yields & CSIF proxy", 
         cmap=new_cmap, cbar_label="Fraction of country-crop pairs with no detected year offset",
         filename="leadlag_pct_map")

    ### fao maize flags
    colors = ["#fdd7de", "red", "darkred"]
    custom_cmap = LinearSegmentedColormap.from_list("pink_to_darkred", colors)
    
    fao_all = pd.read_csv("./data/fao_flag_pct.csv")
    fao_maize = fao_all.query("cropname == 'Maize'")
    colors = ["#fdd7de", "red", "darkred"]
    custom_cmap = LinearSegmentedColormap.from_list("pink_to_darkred", colors)
    
    plot_map(fao_maize, column="flag_pct", title="FAO-reported maize yield flags", cmap=custom_cmap, 
     cbar_label=None, 
     edgecolor="gray", vmin=0, vmax=1,
     legend_kwds={'label': "Fraction of FAO-flagged maize yield observations",
                  'ticks': [0, 0.25, 0.5, 0.75, 1.0], "orientation": "horizontal", 'pad':-0.01},
     missing_kwds={'color': 'white'},
     filename="fao_flags_maize_map")
    
    ## fao flag analysis - all crops averaged

    fao_avg = fao_all.groupby('iso_a3')['flag_pct'].mean().reset_index()
    
    plot_map(fao_avg, column="flag_pct", title="FAO-reported yield flags (all crops)", cmap=custom_cmap, 
     cbar_label=None, 
     edgecolor="gray", vmin=0, vmax=1,
     legend_kwds={'label': "Fraction of FAO-flagged yield observations (averaged across crops)",
                  'ticks': [0, 0.25, 0.5, 0.75, 1.0], "orientation": "horizontal", 'pad':-0.01},
     missing_kwds={'color': 'white'},
     filename="fao_flags_allcrops_map")