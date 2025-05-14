import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# Datasets explore.")
    return (mo,)


@app.cell
def _(mo):
    import polars as pl
    import os
    from tqdm.auto import tqdm

    mo.md("## Imports")
    return os, pl, tqdm


@app.cell
def _(mo):
    from utils.datasets import HydroStaticFeaturesFiles
    from utils.datasets import MeteoSeriesFeaturesFiles
    from utils.datasets import HydroFiles

    mo.md("""## Iterable class for meteo features""")
    return HydroFiles, HydroStaticFeaturesFiles, MeteoSeriesFeaturesFiles


@app.cell
def _(HydroFiles, MeteoSeriesFeaturesFiles, pl):
    from tabulate import tabulate

    def describe_dataset(dataset):
        file = next(iter(dataset))
        lf = pl.read_csv(file).lazy() if isinstance(file, str) else file
        schema = lf.collect_schema()
        schema_table = [{"Column": col, "Type": str(dtype)} for col, dtype in schema.items()]
        return tabulate(schema_table, headers="keys", tablefmt="github")

    msff = MeteoSeriesFeaturesFiles()
    hf = HydroFiles()

    datasets_field_types = \
        "## Review of datasets\n" \
        + "### Meteo Features\n"  \
        + describe_dataset(msff)   \
        + "\n\n" \
        + "### Hydrologic data\n" \
        + describe_dataset(hf)
    return datasets_field_types, describe_dataset, hf, msff, tabulate


@app.cell
def _(datasets_field_types, mo):
    mo.md(datasets_field_types)
    return


@app.cell
def _(hf):
    next(iter(hf)).collect()
    return


@app.cell
def _():
    # from utils.datasets import plot_meteo_series

    # for file in msf:
    #     plot_meteo_series(file)
    #     break
    return


if __name__ == "__main__":
    app.run()
