import geopandas as gpd
import pandas as pd
import requests

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

def clean_fao_flags():
    fao = (pd.read_csv("./data/faostat_all_flags_raw.csv")[lambda df: df['Flag Description'] == "Estimated value"]
       .query("Year > 1999")[['Area Code (ISO3)', 'Year', 'Flag Description', 'Item']].rename(columns={'Area Code (ISO3)': "iso_a3", 'Item': 'item'}))
    country_key = pd.read_csv("./data/country_key.csv")
    fao = fao.merge(country_key, how="left", on="iso_a3").drop(columns={"lowres", "wb"})
    cropkey = pd.read_pickle("./data/calendar_fao_cropkey.pkl")
    fao = fao.merge(cropkey[['item', 'cropname']], on='item', how="left")
    #fao.to_csv("faostat_all_flags.csv",index=False)




__all__ = ["clean_map_gpd"]

