import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import xgboost.dask as dxgb
from dask.distributed import Client, wait
from newutils.data.Dataset import Dataset
from newutils.math import pseudo_mape_obj


class Experiment:
    def __init__(
        self, client: Client, artifacts_path: Path | str, dataset: Dataset = None
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.client = client
        self.artifacts_path = Path(artifacts_path)
        self.dataset = Dataset(self.client) if dataset is None else dataset
        self.model_paths = list()
        self.df = None

    def prepare_experiment(self, num_load: int = 8) -> None:
        self.logger.info("Prepare experiment.")
        self.dataset.create(num_load)
        self.X_train, self.y_train, _, _ = self.dataset.train_test_Xy_split()
        self.logger.warning("DEBUG: Use only one feature to predict.")
        self.y_train = self.y_train[Dataset.TARGET]
        self.dtrain = self.dataset.create_dmatrix(self.X_train, self.y_train)

    def train_model(self, params: Dict[str, Any]) -> None:
        self.logger.info("Start model training.")
        self.train_results = dxgb.train(
            self.client,
            params,
            self.dtrain,
            evals=[(self.dtrain, "train")],
            num_boost_round=params["num_boost_round"],
            verbose_eval=params["verbose_eval"],
            obj=pseudo_mape_obj,
        )

    def export_model(self) -> None:
        self.logger.info("Save the result.")
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H%M%S")
        self.model_paths.append(self.artifacts_path / f"model{timestamp}.json")
        self.train_results["booster"].save_model(self.model_paths[-1])

    def make(self, params: Dict[str, Any]) -> None:
        self.prepare_experiment(params["num_load"])
        self.train_model(params)
        self.export_model()
