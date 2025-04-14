from .CommonImports import *


class MeteoSeriesFeaturesFiles:
    def __init__(self, dir: Path = DIR_METEO_SERIES_FEATURES):
        self.dir = dir
        self.file_ids = [int(file.stem) for file in self.dir.iterdir() if file.suffix == ".nc"]
        self._current = 0
        self._end = len(self.file_ids)

    def get_ids(self) -> List[str]:
        return self.file_ids.copy()

    def _load_id(self, id: int) -> pl.LazyFrame:
        df = (
            xr.open_dataset(self.dir / (str(id) + ".nc"))
            .to_dataframe()
            .reset_index()
        )
        return pl.from_pandas(df, schema_overrides={
            "gauge_id": pl.Int64,
            "date": pl.Date
        }).lazy()

    def __len__(self):
        return self._end

    def __iter__(self) -> Iterator[pl.LazyFrame]:
        for id in self.file_ids:
            yield self._load_id(id)

    def __getitem__(self, gauge_id: int) -> pl.LazyFrame:
        if gauge_id not in self.file_ids:
            raise KeyError(f"Gauge ID {gauge_id} not found in dataset.")
        return self._load_id(gauge_id)
