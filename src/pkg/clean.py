import geopandas as gpd
import pandas as pd
import requests
import numpy as np
import glob
import xarray as xr
import os

def first_clean(path):
    mylist = [f for f in glob.glob(path+"/*")] #/Volumes/common/data/DataByCountry/*
    end_fao = '/fao_data.nc'
    end_cal = '/crop_calendar_data_sage.nc'
    end_sif = '/csif_data.nc'
    end_harvarea = "/harvest_area_fraction_data_sage_all.nc"

    myends = ["end_cal", "end_sif", "end_harvarea"]      

    remove=[]

    for i in mylist: 
        for end in myends:        
            if os.path.isfile(str(i) + eval(end)):
                pass
            else:        
                remove.append(i)
                remove = [*set(remove)]

    allcountries = set(mylist).difference(set(remove))
    return(allcountries)


def clean_map_gpd():
    country_key = pd.read_csv("./data/country_key.csv")[['lowres', 'country']]
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries = countries.rename({"name":"lowres"}, axis=1)
    countries = countries[countries['continent']!= "Antarctica"]
    countries.loc[countries.lowres=="France", 'iso_a3'] = "FRA"
    countries.loc[countries.lowres=="Norway", 'iso_a3'] = "NOR"
    mollweide_proj = '+proj=moll +lon_0=0'
    countries = countries.to_crs(mollweide_proj).merge(country_key, on="lowres")
    return countries

def clean_fao_flags():
    fao = (pd.read_csv("./data/faostat_all_flags_raw.csv")[lambda df: df['Flag Description'] == "Estimated value"]
       .query("Year > 1999")[['Area Code (ISO3)', 'Year', 'Flag Description', 'Item']].rename(columns={'Area Code (ISO3)': "iso_a3", 'Item': 'item'}))
    country_key = pd.read_csv("./data/country_key.csv")
    fao = fao.merge(country_key, how="left", on="iso_a3").drop(columns={"lowres", "wb"})
    cropkey = pd.read_pickle("./data/calendar_fao_cropkey.pkl")
    fao = fao.merge(cropkey[['item', 'cropname']], on='item', how="left")
    #fao.to_csv("faostat_all_flags.csv",index=False)


def fetch_clean_wb():
    url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD"
    params = {"date": "2000:2023","format": "json", "per_page": 20000}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            records = data[1] 
            gdp_df = pd.DataFrame(records)
            gdp_df = gdp_df[["countryiso3code", "date", "value"]]
            gdp_df.columns = ["iso_a3", "year", "value"]
            gdp_df = gdp_df.groupby('iso_a3').mean('value').reset_index()
        else:
            print("No data found in the API response.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

    country_key = pd.read_csv("./data/country_key.csv")
    gdp_df = gdp_df.merge(country_key, how="left", on="iso_a3").drop(columns={"lowres", "fao"})
    #gdp_df.to_csv("wb_gdp_per_cap.csv",index=False)

    inc_class = pd.read_csv("./data/wb_classification_raw.csv", index_col=None).rename(columns={'country': 'wb'})
    key = pd.read_csv("./data/country_key.csv")[['country', 'wb']]
    inc_class = inc_class.merge(key, how="left", on="wb")
    #inc_class.to_csv("./data/wb_classification.csv",index=False)
    return gdp_df, inc_class


def fetch_ag_covariates(start):
    key = pd.read_pickle("./data/calendar_fao_cropkey.pkl")
    idx = np.unique(key['harv_area_idx'].values)

    ## harvested area
    end_harvarea = "/harvest_area_fraction_data_sage_all.nc"
    harvarea = xr.open_dataset(start+end_harvarea)['harvest_area_frac_nearest']
    harvarea['location']= harvarea['location']+0
    harvarea['crop'] = harvarea['crop']+0
    harvarea = harvarea.sel(crop=idx).where(harvarea>0)
    counts = harvarea.groupby('crop').count(...)
    counts = counts.where(counts>0, drop=True)['crop'].values
    if counts.size==0: 
        return
    
    wanted = key[key['harv_area_idx'].isin(counts)]['crop'].values
    harvarea = harvarea.rename({"crop": "harv_area_idx"})
    harvarea = harvarea.where(harvarea.harv_area_idx.isin(counts), drop=True)
    total_harvarea = harvarea.groupby('harv_area_idx').sum('location')
    total_harvarea = total_harvarea.to_dataframe().reset_index().rename(columns={"harvest_area_frac_nearest": "total_harvarea"})
    avg_farmsize = harvarea.where(harvarea.harv_area_idx.isin(counts), drop=True).groupby('harv_area_idx').mean('location')
    avg_farmsize = avg_farmsize.to_dataframe().reset_index().rename(columns={"harvest_area_frac_nearest": "avg_farmsize"})
    areas = total_harvarea.merge(avg_farmsize)

    ### modis cropland types
    end_landtype = "/cropland_fraction_data_modis.nc"
    cropland_frac = xr.open_dataset(start+end_landtype)['cropland_fraction'].mean('year')
    cfrac = pd.DataFrame(xr.merge([cropland_frac, harvarea]).to_dataframe().reset_index().dropna().groupby('harv_area_idx').mean()['cropland_fraction']).reset_index()

    ### majority crop for each country, cropland frac only
    locindex = xr.open_dataset(start+end_landtype)[['lat', 'lon', "location"]].to_pandas().reset_index()
    modis_coords = xr.open_dataset(start+end_landtype)[['lat', 'lon']].set_index(heya=["lat", "lon"]).unstack("heya")
    maj = xr.open_dataset("./data/m2018_majority.nc")["Band1"]
    majcrop = maj.reindex_like(modis_coords, method="nearest")
    majcropdf = majcrop.to_dataframe().reset_index()
    majcropdf = majcropdf.merge(locindex)
    majxr = majcropdf.set_index("location").drop(columns=['lat', 'lon']).to_xarray()
    df= pd.DataFrame(xr.merge([harvarea, majxr]).to_dataframe().reset_index().dropna()[['harv_area_idx', 'Band1']].groupby("harv_area_idx").Band1.value_counts()).rename(columns={"Band1": "counts"}).reset_index()
    majdf = df.loc[df.groupby("harv_area_idx", group_keys=False)['counts'].nlargest(2).index]
    majdf['majcrop_rank']= majdf.groupby('harv_area_idx')['counts'].transform(lambda x: x == x.max()).astype(int).replace(0, 2)
    majdf = majdf.rename(columns={"Band1": "majcrop"}).drop(columns="counts")

    ### putting it all together
    alltog = areas.merge(cfrac).merge(majdf).merge(key[["crop", 'harv_area_idx']], how="left").drop(columns=["harv_area_idx"])
    alltog['country']= start.rsplit('/',1)[1]
    return(alltog)


def flexible_merge(left, right):
    keys = ["iso_a3"]
    if "cropname" in left.columns and "cropname" in right.columns:
        keys.append("cropname")
    return pd.merge(left, right, on=keys, how="outer")



__all__ = ["clean_map_gpd", "first_clean", "fetch_ag_covariates", "flexible_merge"]

