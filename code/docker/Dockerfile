FROM daskdev/dask:2025.4.1-py3.11

# Install Poetry and your Python dependencies
RUN pip install poetry
RUN conda install -y \
    'dask[complete]==2025.4.1' \
    'distributed==2025.4.1' \
    'geopandas>=1.0.1,<2.0.0' \
    'scipy>=1.15.2,<2.0.0' \
    'matplotlib>=3.10.1,<4.0.0' \
    'tqdm>=4.67.1,<5.0.0' \
    'xarray>=2025.1.2,<2026.0.0' \
    'netcdf4>=1.7.2,<2.0.0' \
    'seaborn>=0.13.2,<0.14.0' \
    'ipykernel>=6.29.5,<7.0.0' \
    'fastcluster>=1.2.6,<2.0.0' \
    'shap>=0.47.1,<0.48.0' \
    'fastparquet>=2024.11.0,<2025.0.0' \
    'bokeh==3.7.2' \
    'panel==1.6.3' \
    'polars>=1.29.0,<2.0.0' \
    'xgboost==2.1.4' \
    'iprogress>=0.4,<0.5' \
    'ipywidgets>=8.1.7,<9.0.0'

WORKDIR /opt/app
COPY pyproject.toml poetry.lock README.md ./
COPY ./src /opt/app/src

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

ENV PYTHONPATH="/opt/app/src:${PYTHONPATH}"