
#import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pandas.api.types import is_numeric_dtype
from .clean import clean_map_gpd

#import os

def annotate_corr(data, color, **kws):
    ax = plt.gca()
    sub_df = data[['csif_log_dt', 'yield_log_dt']].dropna()
    if len(sub_df) >= 2:
        r, _ = stats.pearsonr(sub_df['csif_log_dt'], sub_df['yield_log_dt'])
        ax.annotate(f"r = {r:.2f}",
            xy=(0.08, 0.90), xycoords=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightyellow'))


def plot_yield_secondary(data, **kwargs):
    data.columns = data.columns.str.lower()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    kwargs.pop("color", None)
    sns.lineplot(data=data, x="year", y="yield_log_dt", ax=ax2, lw=2, ci=None, color="black")
    ax2.set_ylabel("")  
    if data['year'].notna().any():
        xmin, xmax = np.nanmin(data['year']), np.nanmax(data['year'])
        ax1.set_xlim(xmin, xmax)
        xticks = np.linspace(round(xmin), round(xmax), num=3, dtype=int)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(t) for t in xticks])


def plot_twin_lines(data, col, col_order, col_wrap, grid_kwargs, plot_kwargs, filename, fl=None):
    data.columns = data.columns.str.lower()

    grid_args = dict(data=data, col=col, 
        col_order=col_order, col_wrap=col_wrap,
        height=4, aspect=1.4,
        sharey=False,despine=False)
    if grid_kwargs:
        grid_args.update(grid_kwargs)

    g = sns.FacetGrid(**grid_args)

    plot_args = dict(x="year", y="csif_log_dt", lw=3, ci=None)
    if plot_kwargs:
        plot_args.update(plot_kwargs)

    g.map_dataframe(sns.lineplot, **plot_args)

    for ax in g.axes.flat:
        ax.tick_params(axis='y', colors='red')
        ax.set_ylabel("", color='red')

    if fl is not None:
        def add_spans(data, **kwargs):
            ax = plt.gca()
            iso = data["country"].iloc[0]   # seaborn intrinsically facets (subsets), so i don't have to worry about data
            years = [year for country, year in fl if country == iso]
            for year in years:
                ax.axvspan(year - 0.5, year + 0.5,
                            facecolor="lightgreen", alpha=0.3)
        g.map_dataframe(add_spans)
    else:
        pass

    g.map_dataframe(annotate_corr)
    g.map_dataframe(plot_yield_secondary)
    g.set_axis_labels("", "")
    g.set_titles(col_template="{col_name}", size=18)

    g.fig.text(0.5, 0.035, 'Year', ha='center', va='center')
    g.fig.text(0.03, 0.5, 'Satellite-derived yields (log, dt)', ha='center', va='center',
            rotation='vertical', color="red")
    g.fig.text(0.985, 0.5, 'Survey-derived yields (log, dt)', ha='center', va='center',
            rotation='-90')

    g.fig.tight_layout(rect=[0.05, 0.06, 0.96, 0.97])
    plt.savefig(f"./plots/{filename}.pdf", bbox_inches='tight')
    plt.show()
    plt.close()


def plot_map(data, column, title, cbar_label, cmap, filename, **kwargs):
    countries = clean_map_gpd()
    merged = countries.merge(data, how="left", on="iso_a3")

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    plot_args = dict(column=column, ax=ax, legend=True, cmap=cmap, 
                     missing_kwds={'color': 'lightgrey'})
    
    if is_numeric_dtype(merged[column]):
        plot_args["legend_kwds"] = {'label': cbar_label,
                        'orientation': "horizontal", 'pad': -0.01}

    plot_args.update(kwargs)
    merged.plot(**plot_args)
    ax.set_title(title)
    ax.set_axis_off()

    if {"sif_coefficient", "pos_neg"} <= set(merged.columns):
        merged.loc[merged.sif_coefficient<1].plot(column="pos_neg", hatch = "//", ax=ax, facecolor="none", edgecolor="white")
    else:
        pass

    plt.savefig(f"./plots/{filename}.pdf", bbox_inches='tight')
    plt.show()
    plt.close()


__all__ = ["annotate_corr", "plot_yield_secondary", "plot_twin_lines", "plot_map"]


