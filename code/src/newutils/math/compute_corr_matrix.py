import pandas as pd
import polars as pl
from tqdm.auto import tqdm


def compute_corr_matrix(lf: pl.LazyFrame, method: str = "pearson") -> pd.DataFrame:
    numeric_columns = lf.select(pl.selectors.numeric()).collect_schema().names()
    corr_dict = {
        col1: [
            lf.select(pl.corr(pl.col(col1), pl.col(col2), method=method))
            .collect()
            .item()
            for col2 in numeric_columns
        ]
        for col1 in tqdm(numeric_columns)
    }

    corr_df = pd.DataFrame(corr_dict, index=numeric_columns)
    return corr_df
