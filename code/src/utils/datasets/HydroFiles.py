from .CommonImports import *


class HydroFiles:
    def __init__(self, dir: Path = DIR_HYDRO_FILES):
        self.dir = dir
        self.filenames = os.listdir(dir)
        self.file_ids = [int(filename.split(".")[0]) for filename in self.filenames]
        self._current = 0
        self._end = len(self.file_ids)

    def get_ids(self):
        return self.file_ids.copy()

    def _load_id(self, id: int) -> pl.LazyFrame:
        filepath = self.dir / f"{id}.csv"
        return pl.scan_csv(
            filepath,
            schema={
                "date": pl.Date,
                "q_cms": pl.Float64,
                "q_mm_day": pl.Float64,
                "lvl_sm": pl.Float64,
                "lvl_mbs": pl.Float64,
            },
        )

    def __len__(self):
        return self._end

    def __iter__(self) -> Iterator[pl.LazyFrame]:
        for gauge_id in self.file_ids:
            yield self._load_id(gauge_id)

    def __getitem__(self, gauge_id: int) -> pl.LazyFrame:
        if gauge_id not in self.file_ids:
            raise KeyError(f"Gauge ID {gauge_id} not found in dataset.")
        return self._load_id(gauge_id)
