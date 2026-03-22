# FAO Agricultural Yield Check

Compares FAO survey-reported crop yields against a CSIF satellite proxy, models what drives agreement between the two, and visualizes results.

## Setup

Requires Python >= 3.9.

```bash
pip install -e .
```

This installs the local `pkg` package and all dependencies.

## Running the pipeline

Run all scripts from the **repository root**:

```bash
python src/scripts/2.YC_model.py
python src/scripts/3.YC_line_plots.py
python src/scripts/4.YC_map_plots.py
python src/scripts/5.teaser_weather_analysis.py
```

### Scripts 0 and 1 (data regeneration only)

`0.generating_csif_proxy.py` and `1.generating_YC_model_df.py` regenerate the intermediate CSVs from raw per-country data that is not included in this repository. **You do not need to run them** — all required CSVs are already provided in `data/`.

### Scripts 2–5

| Script | Description |
|--------|-------------|
| `2.YC_model.py` | Fits a mixed-effects model predicting survey–satellite R² and outputs a LaTeX table |
| `3.YC_line_plots.py` | Time-series plots of survey vs. satellite yields (USA crops, global maize) |
| `4.YC_map_plots.py` | Choropleth maps of R², GDP, lead/lag, and FAO flags |
| `5.teaser_weather_analysis.py` | Regresses yields on soil moisture and tmax; produces summary tables and maps |

## Repository structure

- `src/pkg/` — Shared utility functions (cleaning, metrics, visualization)
- `src/scripts/` — Analysis pipeline (run in order, starting from step 2)
- `data/` — Finished data products used in analysis
- `notebook/` — Exploratory notebooks
- `plots/` — All plots included in manuscript
