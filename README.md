# Data leakage in the cross-validation of machine learning water quality models

Code for the paper "Data leakage in the cross-validation of machine learning water quality models: near-duplicate
samples and spatial proximity".

## Data

| Dataset | Source |
|---|---|
| GLORIA | Lehmann et al. (2022), https://doi.org/10.1594/PANGAEA.948492 |
| Caravan-Qual (lite) v1.1 | Jones et al. (2026), https://doi.org/10.5281/zenodo.21337532 |
| USGS SSC | Nookala et al. (2026), https://doi.org/10.5281/zenodo.22298970 |
| CoastDOM v1 | Lønborg et al. (2023), https://doi.org/10.1594/PANGAEA.964012 |

The datasets (CC BY 4.0) are not included. `python -m swq.download` downloads them into `data/` (about 2.2 GB) and
checks them against the versions used in the paper. The map outlines in `maps/` are simplified from Natural Earth (public domain).

## Usage

Python 3.12:

    pip install -r requirements.txt
    python -m swq.download
    python -m swq.prepare
    python -m swq.paper run --workers 2
    python -m swq.paper analyze
    python -m swq.figures --tables output/tables

`run` fits LightGBM, ridge regression, kNN, random forest, MLP and the location-only model for the 13 targets under
random, blocked and station-wise CV; finished units are skipped when it is repeated. With two workers it takes about
5 hours on a laptop. `analyze` writes the tables to
`output/tables/`, and `figures` writes Figs. 1-7, Tables 1-3, Tables S1-S2 and the graphical abstract to `output/publication/`.

`paper_results/tables/` holds the numbers in the paper; without `--tables`, `figures` uses them. `paper_results/folds/`
gives the fold of every sample for each scheme and seed.

## License

MIT. The datasets keep their own licenses.
