from .CommonImports import *


class HydroStaticFeaturesFiles:
    def __load_dataframe(self, filepath):
        raw_df = pl.scan_csv(filepath, schema_overrides={"gauge_id": pl.Int64})
        return raw_df.with_columns([
            pl.col(col).cast(pl.Float64)
            for col in raw_df.collect_schema().names()
            if col != "gauge_id"
        ]).collect().lazy()

    def __init__(self, filepath: Path = DIR_HYDRO_STATIC_FEATURES):
        self.dataframe = self.__load_dataframe(filepath)

        self.ids = (
            self.dataframe.select(pl.col("gauge_id").unique()).collect()
            .get_column("gauge_id")
            .cast(pl.Int64)
            .to_list()
        )
        self._end = len(self.ids)

    def get_ids(self) -> List[int]:
        return self.ids.copy()

    def __len__(self) -> int:
        return self._end

    def __iter__(self) -> Iterator[tuple[int, pl.LazyFrame]]:
        for gauge_id in self.ids:
            yield gauge_id, self.dataframe.filter(pl.col("gauge_id") == gauge_id)

    def __getitem__(self, gauge_id: int) -> pl.LazyFrame:
        if gauge_id not in self.ids:
            raise KeyError(f"Gauge ID {gauge_id} not found in dataset.")
        return self.dataframe.filter(
            pl.col("gauge_id") == gauge_id
        ).drop(pl.col("gauge_id"))
