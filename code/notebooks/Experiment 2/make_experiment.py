from dask.distributed import Client, LocalCluster, wait
import xgboost.dask as dxgb
import dask.dataframe as dd
from datetime import datetime

from newutils.math import pseudo_mape_obj
from newutils.data.datasets import (
    add_lags,
    load_train_subset,
    load_train_subset_fixed_per_file,
)

from oldutils.types import TimeRange
from oldutils.datasets import (
    STATIC_FEATURES,
    HydroStaticFeaturesFiles,
)


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

    def _add_lags(self):
        self.lags_columns = [
            "prcp",
            "t_max",
            "t_min",
            "t_mean",
            "lvl_sm",
        ]  # + ["q_mm_day"]
        lags = range(1, self.HORIZON_HISTORY + 1)
        df = add_lags(df, self.lags_columns, lags)
        target_lags = range(1, self.HORIZON_FORECAST + 1)
        df = add_lags(df, self.targets, target_lags, self.targets)
        self.targets.extend(self.TARGETS)
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _load_dataset(self, num_load=8):
        self.df = load_train_subset(num_load)
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _clear_dataset(self):
        self.df = self.df.dropna()
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _add_static_features(self):
        hsff = (
            HydroStaticFeaturesFiles()
            .dataframe.collect()
            .to_pandas()[STATIC_FEATURES + ["gauge_id"]]
        )
        hsff = dd.from_pandas(hsff)
        self.df = self.df.merge(hsff, how="left", on="gauge_id")
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _repartition(self, partition_size="10MB"):
        self.df = self.df.repartition(partition_size=partition_size)
        self.df = self.client.persist(self.df)
        wait(self.df)

    def _create_dataset(self, num_load=8):
        self._load_dataset(num_load)
        self._add_lags()
        self._repartition()

        self._clear_dataset()
        self._add_static_features()
        self._repartition()

    def _train_test_Xy_split(self):
        X = self.df.drop(columns=set(self.lags_columns + self.targets + ["date"]))
        y = self.df[self.targets]
        X, y = self.client.persist([X, y])
        wait([X, y])
        return X, y, None, None

    def _create_dmatrix(self, X, y):
        dmatr = dxgb.DaskDMatrix(self.client, X, y)
        return dmatr

    def _prepare_experiment(self, num_load=8):
        self._create_dataset(num_load)
        X_train, y_train, _, _ = self._train_test_Xy_split()
        self.dtrain = self._create_dmatr(X_train, y_train)

    def _train_model(self, params):
        self.train_results = dxgb.train(
            self.client,
            params,
            self.dtrain,
            num_boost_round=1000,
            verbose_eval=50,
            evals=[(self.dtrain, "train")],
            obj=pseudo_mape_obj,
        )

    def _export_model(self):
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H%M%S")
        self.model_paths.append(self.artifacts_path / f"model{timestamp}.json")
        self.train_results["booster"].save_model(self.model_paths[-1])

    def make_experiment(self, params, num_load=8):
        self._prepare_experiment(num_load)
        self._train_model(params)
        self._export_model()
