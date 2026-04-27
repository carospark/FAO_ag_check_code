"""
Supplementary figures:
  S1) Per-crop satellite adj-R^2 maps (maize, sorghum, wheat, cassava, potatoes)
      addresses concern that the spatial pattern in Figure 4a could be driven by
      one or two crops.

  S2) Histogram of country-crop rho^2_VS values, optionally split by GDP tercile.
      addresses the manuscript's qualitative description of the rho^2 distribution.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pkg import detrend_group
from pkg.clean import clean_map_gpd


# %%
# ---------------- S1: per-crop satellite adj R^2 maps ----------------

CROPS = ["Maize", "Sorghum", "Wheat", "Cassava", "Potatoes"]


def safe_adjr2(group):
    if len(group.dropna(subset=["csif_log_dt", "sm_dt", "tmax_dt"])) <= 4:
        return np.nan
    try:
        return smf.ols("csif_log_dt ~ sm_dt + tmax_dt", data=group).fit().rsquared_adj
    except Exception:
        return np.nan


yields = pd.read_csv("./data/yields.csv")
yields = yields.loc[:, ~yields.columns.str.contains("_lag|_lead|fao_idx|gridcells", regex=True)]
sm_tmax = pd.read_csv("./data/sm_tmax.csv")
yc = yields.merge(sm_tmax, how="left", on=["year", "cropname", "country"])
yc = detrend_group(yc, "sm_og", "sm_dt")
yc = detrend_group(yc, "tmax_og", "tmax_dt")

counts = yc[["country", "cropname"]].value_counts()
keep_idx = counts[counts > 10].index
yc10 = yc.set_index(["country", "cropname"]).loc[keep_idx].reset_index()

per_crop_country = (
    yc10[yc10["cropname"].isin(CROPS)]
    .groupby(["cropname", "country"])
    .apply(safe_adjr2)
    .rename("adj_r2")
    .reset_index()
)

country_key = pd.read_csv("./data/country_key.csv")[["iso_a3", "country"]]
per_crop_country = per_crop_country.merge(country_key, how="left", on="country").dropna(subset=["adj_r2"])

# %%
countries_geo = clean_map_gpd()

fig, axes = plt.subplots(2, 3, figsize=(24, 12))
axes = axes.flatten()
vmin, vmax = 0, 1
cmap = "plasma_r"

for ax, crop in zip(axes, CROPS):
    sub = per_crop_country[per_crop_country["cropname"] == crop]
    merged = countries_geo.merge(sub[["iso_a3", "adj_r2"]], how="left", on="iso_a3")
    merged.plot(
        column="adj_r2", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        missing_kwds={"color": "lightgrey"}, legend=False,
    )
    ax.set_title(crop, fontsize=20)
    ax.set_axis_off()

# hide the empty 6th panel and add a shared horizontal colorbar in its slot
axes[-1].set_axis_off()
sm_obj = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar_ax = fig.add_axes([0.70, 0.30, 0.20, 0.025])
cbar = fig.colorbar(sm_obj, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Adjusted $R^2$ (satellite weather model)", fontsize=14)

fig.suptitle(
    "Per-crop satellite yield variability explained by SM and TMAX",
    fontsize=22, y=0.97,
)
plt.savefig("./plots/figS1_per_crop_satellite_adjr2.pdf", bbox_inches="tight")
plt.show()
plt.close()


# %%
# ---------------- S2: histogram of rho^2_VS ----------------

df = pd.read_csv("./data/yield_comparison_df.csv")
gdp = pd.read_csv("./data/wb_gdp_per_cap.csv").rename({"value": "gdp"}, axis=1)
gdp = gdp.groupby(["iso_a3"]).agg(avg_gdp=("gdp", "mean")).reset_index()

df = df.merge(gdp, on="iso_a3", how="left", suffixes=("", "_gdp"))
gdp_col = "avg_gdp_gdp" if "avg_gdp_gdp" in df.columns else "avg_gdp"
df = df.dropna(subset=["r2", gdp_col])

# Tercile split on country-level avg GDP
country_gdp = df[["iso_a3", gdp_col]].drop_duplicates()
country_gdp["gdp_tercile"] = pd.qcut(
    country_gdp[gdp_col], q=3, labels=["Low GDP", "Mid GDP", "High GDP"]
)
df = df.merge(country_gdp[["iso_a3", "gdp_tercile"]], on="iso_a3", how="left")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: overall histogram
ax = axes[0]
ax.hist(df["r2"], bins=40, color="steelblue", edgecolor="white")
ax.axvline(df["r2"].mean(), color="black", linestyle="--", linewidth=1.5,
           label=f"mean = {df['r2'].mean():.2f}")
ax.axvline(df["r2"].median(), color="black", linestyle=":", linewidth=1.5,
           label=f"median = {df['r2'].median():.2f}")
ax.set_xlabel(r"$\rho^2_{VS}$ (country--crop)")
ax.set_ylabel("Count")
ax.set_title(f"All country--crop pairs (N = {len(df)})")
ax.legend(frameon=False)

# Panel B: split by GDP tercile
ax = axes[1]
colors = {"Low GDP": "firebrick", "Mid GDP": "goldenrod", "High GDP": "steelblue"}
bins = np.linspace(0, df["r2"].max(), 30)
for label in ["Low GDP", "Mid GDP", "High GDP"]:
    sub = df[df["gdp_tercile"] == label]["r2"]
    ax.hist(sub, bins=bins, alpha=0.55, color=colors[label],
            label=f"{label} (mean = {sub.mean():.2f})")
ax.set_xlabel(r"$\rho^2_{VS}$ (country--crop)")
ax.set_ylabel("Count")
ax.set_title("By country GDP tercile")
ax.legend(frameon=False)

fig.tight_layout()
plt.savefig("./plots/figS2_rho2_distribution.pdf", bbox_inches="tight")
plt.show()
plt.close()


# %%
# ---------------- S3: per-crop rho^2_VS maps (all 19 crops) ----------------
# Per-crop version of Figure 3a, which shows the across-crops max of rho^2 by country.

rho2 = pd.read_csv("./data/yield_comparison_df.csv")[["iso_a3", "cropname", "r2"]]

# For each (country, crop), keep the best lag's r2 (matches Figure 3a's "max" treatment)
rho2 = rho2.groupby(["iso_a3", "cropname"], as_index=False)["r2"].max()

ALL_CROPS = sorted(rho2["cropname"].unique())  # 19 crops

n_cols, n_rows = 4, 5  # 20 panels for 19 crops + 1 colorbar slot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 22))
axes = axes.flatten()
vmin, vmax = 0, 1
cmap = "viridis"

for ax, crop in zip(axes, ALL_CROPS):
    sub = rho2[rho2["cropname"] == crop]
    merged = countries_geo.merge(sub[["iso_a3", "r2"]], how="left", on="iso_a3")
    merged.plot(
        column="r2", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        missing_kwds={"color": "lightgrey"}, legend=False,
    )
    ax.set_title(crop, fontsize=16)
    ax.set_axis_off()

# hide leftover empty panels and put a horizontal colorbar in the last slot
for ax in axes[len(ALL_CROPS):]:
    ax.set_axis_off()
sm_obj = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar_ax = fig.add_axes([0.78, 0.08, 0.18, 0.018])
cbar = fig.colorbar(sm_obj, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$\rho^2_{VS}$ (FAO yields vs. CSIF proxy)", fontsize=14)

fig.suptitle(
    r"Per-crop survey--satellite correspondence ($\rho^2_{VS}$) by country",
    fontsize=22, y=0.995,
)
plt.savefig("./plots/figS3_per_crop_rho2_maps.pdf", bbox_inches="tight")
plt.show()
plt.close()
