from oldutils.datasets import ARTIFACTS_FOLDER
import polars as pl
import dask.dataframe as dd

PATH_MERGED_DATASETS = ARTIFACTS_FOLDER / "merged_datasets"
FILENAME_TRAIN_IDS = "train_file_ids.csv"
COL_GAUGE_ID = "gauge_id"


def load_train_subset():
    """Load train subset based on 'Prepare Dataserts.ipynb' results

    Returns:
        dask.dataframe: dataframe of readed files.
    """
    train_df = pl.read_csv(PATH_MERGED_DATASETS / FILENAME_TRAIN_IDS).sample(10)
    file_ids = train_df["file_id"].to_list()
    paths = [PATH_MERGED_DATASETS / f"{file_id}.parquet" for file_id in file_ids]
    return dd.read_parquet(paths).set_index(COL_GAUGE_ID)


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
