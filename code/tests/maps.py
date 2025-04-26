import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""#""")
    return


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    from keplergl import KeplerGl
    return KeplerGl, gpd, mo


@app.cell
def _():
    files = dict()
    return (files,)


@app.cell
def _(files, gpd):
    files["GTS_river_gauges"] = gpd.read_file("data/GTS_river_gauges.gpkg")
    return


@app.cell
def _(files, gpd):
    files["gts_rivers_ws"] = gpd.read_file("data/gts_rivers_ws.gpkg")
    return


@app.cell
def _(KeplerGl, files, mo):
    map_1 = KeplerGl(height=500)

    map_1.add_data(data=files["GTS_river_gauges"], name='My GeoData')
    # map_1.add_data(data=files["gts_rivers_ws"], name="Polygon Layer")

    map_1.save_to_html(file_name="test.html")

    mo.iframe(html=map_1._repr_html_())
    return (map_1,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
