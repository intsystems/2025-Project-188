import matplotlib.pyplot as plt
import seaborn as sns


class TSPlotStyle:
    def __init__(self, context="paper", style="darkgrid", palette="Set2"):
        self._ctx = sns.plotting_context(context)
        self._sty = sns.axes_style(style)
        self._pal = sns.color_palette(palette)

    def __enter__(self):
        self._ctx.__enter__()
        self._sty.__enter__()
        self._pal.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pal.__exit__(exc_type, exc_val, exc_tb)
        self._sty.__exit__(exc_type, exc_val, exc_tb)
        self._ctx.__exit__(exc_type, exc_val, exc_tb)


class TSPlotter:
    TITLE_FONTDICT = {
        "size": 16,
        "weight": "bold",
    }

    @staticmethod
    def plot_ts(dataframe, column, ax, alpha=1):
        dataframe[column].plot(ax=ax, alpha=alpha)

    @staticmethod
    def _create_figure():
        plt.figure(figsize=(10, 5))

    @staticmethod
    def _set_title(title):
        plt.title(title, fontdict=TSPlotter.TITLE_FONTDICT)

    @staticmethod
    def _finish_plot():
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_precipitation(dataframe):
        with TSPlotStyle():
            TSPlotter._create_figure()
            TSPlotter._set_title("Precipitation")
            TSPlotter.plot_ts(dataframe, "prcp", plt.gca())
            TSPlotter._finish_plot()

    @staticmethod
    def plot_temperatures(dataframe):
        with TSPlotStyle():
            TSPlotter._create_figure()
            TSPlotter._set_title("Temperatures")
            TSPlotter.plot_ts(dataframe, "t_max", plt.gca(), alpha=0.7)
            TSPlotter.plot_ts(dataframe, "t_mean", plt.gca(), alpha=0.7)
            TSPlotter.plot_ts(dataframe, "t_min", plt.gca(), alpha=0.7)
            TSPlotter._finish_plot()

    @staticmethod
    def plot_daily_runoff(dataframe):
        with TSPlotStyle():
            TSPlotter._create_figure()
            TSPlotter._set_title("Daily depth of runoff")
            TSPlotter.plot_ts(dataframe, "q_mm_day", plt.gca())
            TSPlotter._finish_plot()

    @staticmethod
    def plot_water_level(dataframe):
        with TSPlotStyle():
            TSPlotter._create_figure()
            TSPlotter._set_title("Water-level measurement")
            TSPlotter.plot_ts(dataframe, "lvl_sm", plt.gca())
            TSPlotter._finish_plot()
