from oldutils.datasets import ARTIFACTS_FOLDER
import polars as pl
import dask
import dask.dataframe as dd
import pandas as pd
import warnings

PATH_MERGED_DATASETS = ARTIFACTS_FOLDER / "merged_datasets"
FILENAME_TRAIN_IDS = "train_file_ids.csv"
COL_GAUGE_ID = "gauge_id"


def read_first_n(path, n=10, expected_columns=None):
    if expected_columns is not None:
        return pl.read_parquet(path, n_rows=n).to_pandas()[expected_columns]
    return pl.read_parquet(path, n_rows=n).to_pandas()


def load_train_subset_fixed_per_file(train_subset=None, n=370):
    """Load train subset based on 'Prepare Dataserts.ipynb' results

    Returns:
        dask.dataframe: dataframe of readed files.
    """
    train_df = pl.read_csv(PATH_MERGED_DATASETS / FILENAME_TRAIN_IDS)
    if train_subset is not None:
        train_df = train_df.sample(train_subset, seed=30)
    else:
        train_subset = len(train_df)
    file_ids = train_df["file_id"].to_list()

    paths = [PATH_MERGED_DATASETS / f"{file_id}.parquet" for file_id in file_ids]
    expected_columns = [
        "date",
        "prcp",
        "t_max",
        "t_mean",
        "t_min",
        "q_mm_day",
        "lvl_sm",
        "gauge_id",
    ]
    delayed_dfs = [
        dask.delayed(read_first_n)(p, n=n, expected_columns=expected_columns)
        for p in paths
    ]

    ddf = dd.from_delayed(delayed_dfs)
    divisions = [file_id for file_id in file_ids] + [max(file_ids) + 1]
    divisions.sort()
    ddf = ddf.set_index(
        COL_GAUGE_ID,
        # divisions=[i for i in range(0, NUM_LOAD * 367, 367)]
        # divisions=[i for i in range(0, max(file_ids) + 1)],
        divisions=divisions,
    )

    return ddf


def get_train_gauge_ids():
    train_df = pl.read_csv(PATH_MERGED_DATASETS / FILENAME_TRAIN_IDS)
    file_ids = train_df["file_id"].to_list()
    return file_ids


def load_train_gauge(gauge_id):
    paths = [PATH_MERGED_DATASETS / f"{gauge_id}.parquet"]
    return dd.read_parquet(paths)


def load_train_subset(train_subset=None):
    """Load train subset based on 'Prepare Dataserts.ipynb' results

    Returns:
        dask.dataframe: dataframe of readed files.
    """
    train_df = pl.read_csv(PATH_MERGED_DATASETS / FILENAME_TRAIN_IDS)
    if train_subset is not None:
        train_df = train_df.sample(train_subset, seed=30)
    file_ids = train_df["file_id"].to_list()
    paths = [PATH_MERGED_DATASETS / f"{file_id}.parquet" for file_id in file_ids]
    return dd.read_parquet(paths)


def _gen_lag_name(column, lag):
    return f"{column}_{lag}"


def _gen_add_lags(column, lags, store_new_columns=None):
    for lag in lags:
        col_name = _gen_lag_name(column, lag)
        if store_new_columns is not None:
            store_new_columns.append(col_name)

    def _add_lags(pdf):
        pdf = pdf.sort_values(by=["date"])
        for lag in lags:
            with warnings.catch_warnings(record=True) as caught_warnings:
                pdf[_gen_lag_name(column, lag)] = pdf[column].shift(lag)
        return pdf

    return _add_lags


def add_lags(df, lags_columns, lags, store_new_columns=None):
    """Add lags features to dataframe per partition.

    Args:
        df (dask.dataframe): dataframe
        lags_columns (list): columns for lagging
        lags (iter): lags for add
        store_new_columns (list, optional): list for store new names. Defaults to None.

    Returns:
        dask.dataframe: result
    """
    for column in lags_columns:
        df = df.map_partitions(_gen_add_lags(column, lags, store_new_columns))
    return df
