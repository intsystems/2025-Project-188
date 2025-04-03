from .CommonImports import *


def plot_meteo_series(df: pl.LazyFrame):
    df = df.collect()
    for column in df.columns:
        if column == "date":
            continue
        plt.figure(figsize=(10, 4))
        sns.lineplot(df, x="date", y=column, ax=plt.gca())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
