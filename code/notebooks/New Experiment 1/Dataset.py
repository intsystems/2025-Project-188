import logging
from functools import partial

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xgboost.dask as dxgb
from config import STATIC_FEATURES
from dask.distributed import Client, LocalCluster, wait
from newutils.data.datasets import (
    load_train_gauges,
    load_train_subset,
    load_train_subset_fixed_per_file,
)
from oldutils.datasets import HydroStaticFeaturesFiles
from oldutils.types import TimeRange


def rolling_mean(pdf: pd.DataFrame, column: str, window: str) -> pd.DataFrame:
    pdf = pdf.sort_values("date").reset_index(drop=True)
    new_col = f"{column}_{window}_mean"
    pdf[new_col] = pdf.rolling(window=window, on="date")[column].mean()
    return pdf


def rolling_sum(pdf: pd.DataFrame, column: str, window: str) -> pd.DataFrame:
    pdf = pdf.sort_values("date").reset_index(drop=True)
    new_col = f"{column}_{window}_sum"
    pdf[new_col] = pdf.rolling(window=window, on="date")[column].sum()
    return pdf


class Dataset:
    HORIZON_HISTORY = TimeRange.YEAR
    HORIZON_FORECAST = TimeRange.WEEK
    LAGS = range(1, HORIZON_HISTORY + 1)
    LAGS_COLUMNS = [
        "prcp",
        "t_max",
        "t_min",
        "t_mean",
        # "lvl_sm",
        # "q_mm_day",
    ]
    TARGET_LAGS = range(1, HORIZON_FORECAST + 1)
    OTHER_X_DROPS = ["date", "gauge_id"]
    TARGET = "q_mm_day"

    def __init__(self, client: Client, gauge_ids=None, static_features=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.client = client
        self.gauge_ids = gauge_ids
        self.static_features = static_features

        self.targets = list()

        self.ts = None
        self.df = None
        self.num_load = 0
        self.hsff = None
        self.X_train = None
        self.y_train = None
        self.repartition_called = False

    def load_static_features(self):
        if self.hsff is None:
            self.logger.info("Load static features.")
            # self.hsff = (
            #     HydroStaticFeaturesFiles()
            #     .dataframe.collect()
            #     .to_pandas()[
            #         (
            #             (STATIC_FEATURES + ["gauge_id"])
            #             if self.static_features is None
            #             else self.static_features
            #         )
            #     ]
            #     .astype({"gauge_id": np.int32})
            # )
            self.hsff = pd.read_csv("data/river_geo_vector.csv").astype(
                {"gauge_id": np.int32}
            )
            self.hsff = dd.from_pandas(self.hsff)
        return self

    def add_lags(self):
        self.logger.info("Generate lags features.")
        if self.repartition_called:
            self.logger.warning("Repartition was called before `add_lags`.")
        for column in Dataset.LAGS_COLUMNS:
            func = rolling_sum if column == "prcp" else rolling_mean
            for window in ["7D", "14D", "21D", "28D", "56D"]:
                new_col = f"{column}_{window}_{('sum' if column=='prcp' else 'mean')}"
                meta = self.df._meta.copy()
                meta[new_col] = np.float64()
                self.df = self.df.map_partitions(
                    partial(func, column=column, window=window), meta=meta
                )
        self.logger.warning("Target lags are disabled.")
        # self.df = add_lags(self.df, [Dataset.TARGET], Dataset.TARGET_LAGS, self.targets)
        self.targets.extend([Dataset.TARGET])
        return self

    def _persist(self, df: dd.DataFrame):
        self.logger.info("Persist.")
        df = self.client.persist(df)
        wait(df)
        return df

    def persist(self):
        self.df = self._persist(self.df)
        return self

    def load_timeseries(self, num_load):
        self.logger.info("Load time series.")
        if self.gauge_ids is None:
            self.df = load_train_subset(num_load)
        else:
            self.logger.warning(
                f"num_load is ignored, loads set of gauge_ids: {self.gauge_ids}"
            )
            self.df = load_train_gauges(self.gauge_ids)
        self.df["date"] = dd.to_datetime(self.df["date"])
        self.df = self.df.astype({"gauge_id": np.int32})
        # TODO: debug here is the sampling
        # self.logger.warning("DEBUG: frac=0.1")
        # self.ts = self.ts.sample(frac=0.1)
        return self

    def dropna(self):
        self.logger.info("Clear dataset from nans.")
        self.df = self.df.dropna()
        return self

    def add_static_features(self):
        self.logger.info("Add static features.")
        self.load_static_features()
        self.df = self.df.merge(self.hsff, how="left", on="gauge_id")
        return self

    def repartition(self, partition_size: str = "100MB"):
        self.repartition_called = True
        self.logger.info(f"Make repartition ({partition_size}).")
        self.df = self.df.repartition(partition_size=partition_size)
        return self

    def recreate(self, num_load: int = 8):
        self.logger.info("Create dataframe.")
        self.load_timeseries(num_load)
        self.persist()

        self.add_lags()
        # self.repartition("100MB")
        self.persist()

        self.dropna()
        self.add_static_features()
        # self.repartition("100MB")
        self.persist()
        self.num_load = num_load

    def create(self, num_load: int = 8):
        if self.num_load != num_load:
            self.recreate(num_load)
        elif self.df is None:
            self.recreate(num_load)
        return self

    def train_test_Xy_split(self):
        if self.df is None:
            self.logger.warning("Call default create.")
            self.create()
        if self.X_train is None or self.y_train is None:
            self.logger.info("Make train test split to X and y.")
            num_parts = self.df.npartitions
            X = self.df.drop(
                columns=set(Dataset.LAGS_COLUMNS + self.targets + Dataset.OTHER_X_DROPS)
            )
            X = X.astype(np.float32)
            # X = X.repartition(npartitions=num_parts)
            y = self.df[self.targets]
            y = y.astype(np.float32)
            # y = y.repartition(npartitions=num_parts)
            X, y = self.client.persist([X, y])
            wait([X, y])
            self.X_train, self.y_train = X, y
        return self.X_train, self.y_train, None, None

    def create_dmatrix(self, X: dd.DataFrame, y: dd.DataFrame) -> dxgb.DaskDMatrix:
        self.logger.info("Create XGBoost DaskDMatrix.")
        return dxgb.DaskDMatrix(self.client, X, y)
