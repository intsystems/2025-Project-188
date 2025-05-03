# ---

# jupyter:

# jupytext:

# text\_representation:

# extension: .py

# format\_name: percent

# format\_version: '1.3'

# jupytext\_version: 1.17.1

# kernelspec:

# display\_name: flood-forecasts-Udo2UZ-X-py3.11

# language: python

# name: python3

# ---

# %% \[markdown]

# # XGBoost Random Forest

# *Experiment 1. Random Forest*

# %%

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear\_output
from tqdm.auto import tqdm

from utils.datasets import (
ARTIFACTS\_FOLDER,
ROOT\_FOLDER,
STATIC\_FEATURES,
HydroFiles,
HydroStaticFeaturesFiles,
MeteoSeriesFeaturesFiles,
)
from utils.types import TimeRange
import polars as pl
import dask.dataframe as dd
from pathlib import Path

# %matplotlib inline

os.chdir(ROOT\_FOLDER)

# %% \[markdown]

# ## Datasets

# %%

PATH\_MERGED\_DATASETS = ARTIFACTS\_FOLDER / "merged\_datasets"
FILENAME\_TRAIN\_IDS = "train\_file\_ids.csv"

# %%

COL\_GAUGE\_ID = "gauge\_id"
TARGETs = \["q\_mm\_day"]

# %%

HORIZON\_HISTORY = TimeRange.YEAR
HORIZON\_FORECAST = TimeRange.WEEK

# %% \[markdown]

# ### Load datasets

# %%

def load\_train\_subset() -> dd:
train\_df = pl.read\_csv(PATH\_MERGED\_DATASETS / FILENAME\_TRAIN\_IDS).sample(10)
file\_ids = train\_df\["file\_id"].to\_list()
paths = \[PATH\_MERGED\_DATASETS / (str(file\_id) + ".parquet") for file\_id in file\_ids]
return dd.read\_parquet(paths)

# %%

df = load\_train\_subset()
df = df.set\_index("gauge\_id")
df.head()

# df = df.repartition(divisions=df.divisions).compute()  # Нет нужды, так как partition по-умолчанию по gauge\_id

# %% \[markdown]

# ### Add lags

# %%

def gen\_lag\_name(column, lag):
return f"{column}\_{lag}"

def gen\_add\_lags(column, lags, store\_new\_columns=None):
for lag in lags:
col\_name = gen\_lag\_name(column, lag)
if store\_new\_columns is not None:
store\_new\_columns.append(col\_name)

```
def add_lags(pdf):
    pdf = pdf.sort_values(by=["date"])
    for lag in lags:
        col_name = gen_lag_name(column, lag)
        pdf[col_name] = pdf[column].shift(lag)
    return pdf

return add_lags
```

# %%

lags = range(1, HORIZON\_HISTORY + 1)
lags\_columns = \["prcp", "t\_max", "t\_min", "t\_mean", "q\_mm\_day", "lvl\_sm"]

# %%

def create\_lags(df, lags\_columns=lags\_columns, lags=lags, store\_new\_columns=None):
for column in lags\_columns:
df = df.map\_partitions(gen\_add\_lags(column, lags, store\_new\_columns))
return df

# %%

df = create\_lags(df)
targets = list()
df = create\_lags(df, TARGETs, range(1, HORIZON\_FORECAST + 1), targets)
targets.extend(TARGETs)

# %%

targets

# %%

df = df.dropna()

# %%

X = df.drop(columns=set(lags\_columns + targets))
y = df\[targets]

X\_train, y\_train = X, y

# %% \[markdown]

# ### Add static features

# %%

hsff = HydroStaticFeaturesFiles()
for i in hsff:
print(i\[1].collect())
break

# %% \[markdown]

# ### Model training

# %%

# ----------------- 0. Cluster -----------------

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
n\_workers=4, threads\_per\_worker=2, memory\_limit="4GB", asynchronous=False
)
client = Client(cluster, asynchronous=False)  # you will see a dashboard at :8787

# %%

from dask.distributed import Client, LocalCluster
from xgboost import dask as dxgb
import xgboost as xgb
import dask.array as da
import numpy as np

params = {
"tree\_method": "hist",  # fast, memory‑frugal
"learning\_rate": 0.1,
"max\_depth": 6,
"subsample": 0.8,
"colsample\_bytree": 0.8,
\# no built‑in objective because we supply one
}

EPS = 1e-6  # smoothing to avoid /0

def mape\_obj(preds: np.ndarray, dmat: xgb.DMatrix):
y = dmat.get\_label()
grad = np.sign(preds - y) / (np.abs(y) + EPS)
hess = EPS / (np.abs(y) + EPS) \*\* 2
return grad, hess

def mape\_eval(preds: np.ndarray, dmat: xgb.DMatrix):
y = dmat.get\_label()
mape = np.mean(np.abs((y - preds) / (y + EPS))) \* 100
return "MAPE", mape

async def main():
cluster = LocalCluster(
n\_workers=4,
threads\_per\_worker=2,
memory\_limit="4GB",
asynchronous=True,  # cluster itself is async
)
client = await Client(cluster, asynchronous=True)

```
# ----- data ----------------------------------------------------------
dtrain = await dxgb.DaskDMatrix(client, X_train.copy(), y_train.copy())

out = await dxgb.train(
    client,
    params,
    dtrain,
    num_boost_round=500,
    obj=mape_obj,
    # maximize=mape_eval,
)
print(out["history"])
```

await main()

# %%

client.close()  # releases port 8787 (or whichever)
cluster.close()

# %%

import gc
gc.collect()
