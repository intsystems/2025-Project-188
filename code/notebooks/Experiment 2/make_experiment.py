import logging
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xgboost.dask as dxgb
from dask.distributed import Client, LocalCluster, wait
from newutils.data.datasets import (
    add_lags,
    load_train_subset,
    load_train_subset_fixed_per_file,
)
from newutils.math import pseudo_mape_obj
from oldutils.datasets import STATIC_FEATURES, HydroStaticFeaturesFiles
from oldutils.types import TimeRange


class Experiment:
    def __init__(self, client, artifacts_path):
        self.client = client
        self.artifacts_path = artifacts_path
        self.HORIZON_HISTORY = TimeRange.YEAR
        self.HORIZON_FORECAST = TimeRange.WEEK
        self.TARGETS = ["q_mm_day"]
        self.targets = list()
        self.model_paths = list()
        self.df = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _add_lags(self):
        self.logger.info("Generate lags features.")
        self.lags_columns = [
            "prcp",
            "t_max",
            "t_min",
            "t_mean",
            "lvl_sm",
        ]  # + ["q_mm_day"]
        lags = range(1, self.HORIZON_HISTORY + 1)
        self.df = add_lags(self.df, self.lags_columns, lags)
        target_lags = range(1, self.HORIZON_FORECAST + 1)
        self.df = add_lags(self.df, self.targets, target_lags, self.targets)
        self.targets.extend(self.TARGETS)
        self._persist()

    def _persist(self):
        self.logger.info("Persist.")
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _load_dataset(self, num_load=8):
        self.logger.info("Load driver data.")
        self.df = load_train_subset(num_load)
        self.df = self.df.sample(frac=0.1)
        # Only for debug! #TODO
        self._repartition("100MB")
        # Remove string above!
        self._persist()

    def _clear_dataset(self):
        self.logger.info("Clear dataset from nans.")
        self.df = self.df.dropna()
        self._persist()

    def _add_static_features(self):
        self.logger.info("Add static features.")
        hsff = (
            HydroStaticFeaturesFiles()
            .dataframe.collect()
            .to_pandas()[STATIC_FEATURES + ["gauge_id"]]
            .astype({"gauge_id": np.int32})
        )
        hsff = dd.from_pandas(hsff)
        self.df = self.df.merge(hsff, how="left", on="gauge_id")
        self._persist()

    def _repartition(self, partition_size="10MB"):
        self.logger.info(f"Make repartition ({partition_size}).")
        self.df = self.df.repartition(partition_size=partition_size)
        self._persist()

    def _create_dataset(self, num_load=8):
        self._load_dataset(num_load)
        self._repartition("50MB")
        self._add_lags()
        self._repartition("50MB")
        self._clear_dataset()
        self._add_static_features()
        self._repartition("5MB")

    def _train_test_Xy_split(self):
        self.logger.info("Make train test split to X and y.")
        X = self.df.drop(columns=set(self.lags_columns + self.targets + ["date"]))
        y = self.df[self.targets]
        X, y = self.client.persist([X, y])
        wait([X, y])
        return X, y, None, None

    def _create_dmatrix(self, X, y):
        self.logger.info("Create xgboost DaskDMatrix.")
        dmatr = dxgb.DaskDMatrix(self.client, X, y)
        return dmatr

    def _prepare_experiment(self, num_load=8):
        self._create_dataset(num_load)
        X_train, y_train, _, _ = self._train_test_Xy_split()
        self.dtrain = self._create_dmatrix(X_train, y_train)

    def _train_model(self, params):
        self.logger.info("Start model training.")
        self.train_results = dxgb.train(
            self.client,
            params,
            self.dtrain,
            evals=[(self.dtrain, "train")],
            obj=pseudo_mape_obj,
        )

    def _export_model(self):
        self.logger.info("Save the result.")
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H%M%S")
        self.model_paths.append(self.artifacts_path / f"model{timestamp}.json")
        self.train_results["booster"].save_model(self.model_paths[-1])

    def make_experiment(self, params, num_load=8):
        self._prepare_experiment(num_load)
        self._train_model(params)
        self._export_model()
