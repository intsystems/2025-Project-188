import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import xgboost.dask as dxgb
from Dataset import Dataset
from newutils.math import pseudo_mape_eval, pseudo_mape_obj
from sklearn.model_selection import train_test_split


class Experiment:
    def train_test_Xy_split(self, test_size: float = 0.2, random_state: int = 30):
        if self.X_train is None or self.y_train is None:
            self.logger.info("Performing train/test split on X and y.")

            drops = set(Dataset.LAGS_COLUMNS + self.targets + Dataset.OTHER_X_DROPS)
            X = self.df.drop(columns=drops)
            y = self.df[self.targets]

            X = X.astype(np.float32)
            y = y.astype(np.float32)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        return self.X_train, self.y_train, self.X_test, self.y_test

    def create_dmatrix(self, X: pd.DataFrame, y: pd.DataFrame) -> xgb.DMatrix:
        self.logger.info("Creating XGBoost DMatrix.")
        data = X.values
        label = y.values
        return xgb.DMatrix(data=data, label=label)

    def __init__(
        self,
        artifacts_path: Path | str,
        dataset: Path = Path("artifacts") / "dataset" / "full_parquet.parquet",
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.df = pd.read_parquet(dataset)
        self.artifacts_path = Path(artifacts_path)
        self.model_paths = list()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.targets = ["q_mm_day"]

    def prepare_experiment(self, num_load: int = 8) -> None:
        self.logger.info("Prepare experiment.")
        self.train_test_Xy_split()
        self.logger.warning("DEBUG: Use only one feature to predict.")
        self.y_train = self.y_train[["q_mm_day"]]
        self.dtrain = self.create_dmatrix(self.X_train, self.y_train)
        self.dtest = self.create_dmatrix(self.X_test, self.y_test)

    def train_model(self, params: Dict[str, Any]) -> None:
        self.logger.info("Start model training.")
        self.train_results = xgb.train(
            params,
            self.dtrain,
            evals=[(self.dtrain, "train"), (self.dtest, "eval")],
            num_boost_round=params["num_boost_round"],
            verbose_eval=params["verbose_eval"],
            # obj=pseudo_mape_obj,
            feval=pseudo_mape_eval,
        )

    def export_model(self) -> None:
        self.logger.info("Save the result.")
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H%M%S")
        self.model_paths.append(self.artifacts_path / f"model{timestamp}.json")
        self.train_results.save_model(self.model_paths[-1])

    def make(self, params: Dict[str, Any]) -> None:
        self.prepare_experiment(params["num_load"])
        self.train_model(params)
        self.export_model()
