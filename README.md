# riskmodels: a library for univariate and bivariate extreme value analysis (and applications to energy procurement)

This library focuses on extreme value models for risk analysis in one and two dimensions. MLE-based and Bayesian generalised Pareto tail models are available for one-dimensional data, while for two-dimensional data, logistic and Gaussian (MLE-based) extremal dependence models are also available. Logistic models are appropriate for data whose extremal occurrences are strongly associated, while a Gaussian model offer an asymptotically independent, but still parametric, copula.

The `powersys` submodule offers utilities for applications in energy procurement, with functionality to model available conventional generation (ACG) as well as to calculate loss of load expectation (LOLE) and expected energy unserved (EEU) indices on a wide set of models; efficient parallel time series based simulation for univariate and bivariate power surpluses is also available. 

## Requirements

This package works with Python >= 3.7

## Documentation

The Github pages of this repo contains the package's documentation

## Quickstart

Because this library grew from research in energy procurement, this example is related to that but the `univariate` and `bivariate` modules are quite general and can be applied to any kind of data. The following example analyses the co-occurrence of high demand net of renewables (this is, demand minus intermittent generation such as wind and solar) in Great Britain's and Denmark's power systems. This can be done to value interconnection in the context of security of supply analysis, for example.

#### Getting the data

The data for this example corresponds roughly to peak (winter) season of 2017-2018, and is openly available online but has to be put together from different places namely [Energinet's API](https://www.energidataservice.dk/), [renewables.ninja](https://www.renewables.ninja/), and [NGESO's website](https://www.nationalgrideso.com/industry-information/industry-data-and-reports/data-finder-and-explorer).

```py
from urllib.request import Request, urlopen
import urllib.parse
import requests 

from time import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import riskmodels.univariate as univar
import riskmodels.bivariate as bivar

def fetch_data():
  def fetch_gb_data():
    wind_url = "https://www.renewables.ninja/country_downloads/GB/ninja_wind_country_GB_current-merra-2_corrected.csv"
    wind_capacity = 13150
    #
    demand_urls = [
      "https://data.nationalgrideso.com/backend/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/2f0f75b8-39c5-46ff-a914-ae38088ed022/download/demanddata_2017.csv",
      "https://data.nationalgrideso.com/backend/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/fcb12133-0db0-4f27-a4a5-1669fd9f6d33/download/demanddata_2018.csv"]
    #
    # function to retrieve reconstructed wind data
    def fetch_wind_data() -> pd.DataFrame:
      """Fetches historic wind generation data reconstructed from atmospheric models
      
      Returns:
          pd.DataFrame: wind data
      """
      print("Fetching GB's wind data..")
      req = Request(wind_url)
      req.add_header('User-Agent', proxy_header)
      content = urlopen(req)
      raw = pd.read_csv(content, header=None)
      df = pd.DataFrame(np.array(raw.iloc[3::,:]), columns = list(raw.iloc[2,:]))
      df["wind_generation"] = wind_capacity * df["national"].astype(np.float32)
      df["time"] = pd.to_datetime(df["time"], utc=True)
      return df.drop(labels=["national", "offshore", "onshore"], axis=1)
    #
    def fetch_demand_data() -> pd.DataFrame:
      """Fetches public demand data for 2017-2018 from NGESO's website
      
      Returns:
          pd.DataFrame: net demand data
      """
      print("Fetching GB's demand data..")
      # fetch data from website
      demand_df = pd.concat([pd.read_csv(url) for url in demand_urls])
      # format time columns
      demand_df["date"] = pd.to_datetime(demand_df["SETTLEMENT_DATE"].astype(str).str.lower(), utc=True)
      demand_df["time"] = demand_df["date"] + pd.to_timedelta([timedelta(days=max((x-1),0.1)/48) for x in demand_df["SETTLEMENT_PERIOD"]])
      demand_df["demand"] = demand_df["ND"]
      return demand_df.filter(items=["time","demand"], axis=1)
    #
    # merge wind and demand and filter peak season of 2017-2018
    df = fetch_demand_data().merge(fetch_wind_data(), on="time")
    #
    df["net_demand"] = df["demand"] - df["wind_generation"]
    #
    return df.filter(items=["time", "net_demand"], axis=1)
  def fetch_dk_data() -> pd.DataFrame:
    """Fetches data for 2017-2018 from the Danish system operator's public API
    
    Returns:
        pd.DataFrame: net demand data
    """
    print("Fetching DK's data..")
    api_url = 'https://data-api.energidataservice.dk/v1/graphql'
    headers = {'content-type': 'application/json'}
    query = """
    {
      electricitybalancenonv(where:{HourUTC:{_gte:\"2017-11-01\"}}, limit: 10000) {
        HourUTC      
        TotalLoad                      
        SolarPower  
        OnshoreWindPower
        OffshoreWindPower
                    
      }
    }
    """
    request = requests.post(api_url, json={'query': query}, headers=headers)
    dk_df = pd.DataFrame(request.json()["data"]["electricitybalancenonv"]).dropna(axis=0)
    dk_df["time"] = pd.to_datetime(dk_df["HourUTC"], utc=True)
    dk_df["net_demand"] = dk_df["TotalLoad"] - dk_df["SolarPower"] - dk_df["OnshoreWindPower"] - dk_df["OffshoreWindPower"]
    return dk_df.filter(items=["time","net_demand"], axis=1)
  #
  gb_df = fetch_gb_data()
  dk_df = fetch_dk_data()
  # leave only peak winter season
  df = gb_df.merge(dk_df, on="time", suffixes=("_gb", "_dk")).query("(time.dt.month >= 11 and time.dt.year == 2017) or (time.dt.month <=3 and time.dt.year == 2018)")
  return df


# fetch data from public APIs
df = fetch_data()
```

#### Univariate extreme value modelling

Empirical distributions are the base on which the package operates, and the `Empirical` classes in both `univariate` and `bivariate` modules provide the main entrypoints. Univariate generalised Pareto models can be fitted in a couple lines.

```py
# set empirical distributions of net demand for both areas
gb_nd, dk_nd = np.array(df["net_demand_gb"]), np.array(df["net_demand_dk"])
# Initialise Empirical distribution objects. They have methods to plot and calculate mean, std, ppf, cdf, among others.
gb_nd_dist, dk_nd_dist = univar.Empirical.from_data(gb_nd), univar.Empirical.from_data(dk_nd)

#look at mean residual life to decide threshold values
q_th = 0.95
gb_nd_dist.plot_mean_residual_life(threshold = gb_nd_dist.ppf(q_th));plt.show()
dk_nd_dist.plot_mean_residual_life(threshold = dk_nd_dist.ppf(q_th));plt.show()
```

Univariate generalised Pareto models can be fitted in one line, and fit diagnostics can be plotted afterwards.

```py
# Fit univariate models for both areas and plot diagnostics
gb_dist_ev = gb_nd_dist.fit_tail_model(threshold=gb_nd_dist.ppf(q_th));gb_dist_ev.plot_diagnostics();plt.plot()
dk_dist_ev = dk_nd_dist.fit_tail_model(threshold=dk_nd_dist.ppf(q_th));dk_dist-ev.plot_diagnostics();plt.plot()
```

#### Bivariate extreme value modelling

Bivariate EV models are built analogous to univariate models. When fitting a bivarate tail model there is a choice between assuming "strong" or "weak" association between extreme co-occurrences across axes \footnote{}. The package implements a Bayesian ratio test, shown below, to help with this decision.
Below, previously fitted univariate models are passed when fitting the bivariate tail model to use these marginal distributions for quantile estimation in the fitting process. If not passed, univariate tail models are fitted to the data.

```py
# instantiate bivariate empirical object with net demand from both areas
bivar_empirical = bivar.Empirical.from_data(np.stack([gb_nd, dk_nd], axis=1))
bivar_empirical.plot();plt.show()

# test for asymptotic dependence and fit corresponding model
r = bivar_empirical.test_asymptotic_dependence(q_th)
if r > 1: # r > 1 suggests strong association between extremes, and fits an asymptotically dependent logistic model
  model = "logistic"
else: # r <= 1 suggests weak association between extremes, and fits an asymptotically independent gaussian model
  model = "gaussian"

bivar_ev_model = bivar_empirical.fit_tail_model(
  model = model,
  quantile_threshold = q_th,
  margin1 = gb_dist_ev,
  margin2 = dk_dist_ev)

```