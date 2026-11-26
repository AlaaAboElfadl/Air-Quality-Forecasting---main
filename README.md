# Air Quality Forecasting

Pattern analysis and pollution prediction using US EPA air quality data. The project explores pollution patterns across US states and cities, and builds machine learning models to forecast the ozone Air Quality Index (**O3 AQI**).

## Dataset

- **Source:** [US Pollution Data (EPA), Kaggle](https://www.kaggle.com/datasets/sogun3/uspollution/data)
- **Pollutants:** NO2, O3, SO2, CO (daily mean, max value, max hour, AQI)
- **Sample used:** 70,922 daily records, 37 stations, 29 states, 2010-2015
- **Target:** `O3 AQI`

## Project Structure

```
Air-Quality-Forecasting/
├── data/
│   ├── raw/                        # original and sampled data
│   ├── processed/                  # cleaned data for analysis
│   └── final/                      # model-ready datasets
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── ml_preprocessing.ipynb
│   ├── supervised_learning.ipynb
│   └── unsupervised_learning.ipynb
├── results/
│   ├── final_model_comparison.csv
│   └── final_test_results_2015.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

```bash
git clone <repository-url>
cd Air-Quality-Forecasting
pip install -r requirements.txt
```

The full Kaggle file is not stored in the repository. The sampled and processed data are included.

## Workflow

Run the notebooks in this order:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `data_understanding` | Explore the raw dataset and its columns |
| 2 | `data_cleaning` | Clean the data (types, missing values, invalid values) |
| 3 | `EDA` | Patterns, maps, seasonality, correlations, anomalies, urban vs rural |
| 4 | `ml_preprocessing` | Time-series preprocessing, lag and rolling features, train/validation/test split |
| 5 | `supervised_learning` | Regression models to forecast O3 AQI |
| 6 | `unsupervised_learning` | Clustering of stations and pollution patterns |

```
raw  ->  processed (cleaning, EDA)  ->  final (features, split)  ->  models  ->  results
```

## Model-Ready Data (`data/final/`)

| File | Description |
|---|---|
| `X_train`, `X_val`, `X_test` | 25 features, not scaled (for tree models) |
| `X_train_std`, `X_val_std`, `X_test_std` | Same features, standardized (for linear models and clustering) |
| `y_train`, `y_val`, `y_test` | Target (`O3 AQI`) |

**Split (chronological):** train 2010-2013, validation 2014, test 2015.

**Features:** lags (1, 7, 30 days) and rolling mean/std (7 days) for the four pollutants, plus `Month`, `Year`, and encoded `Region`, `area_type`, `Season`.

**Notes**
- Do not use same-day pollutant columns as features (data leakage).
- Fit any scaler, encoder, or imputer on the training set only.
- Some lag features contain NaN (start of each station and after long gaps).

## Results

Saved in `results/`:

- `final_model_comparison.csv`: comparison of the trained models.
- `final_test_results_2015.csv`: final evaluation on the 2015 test set.

## Limitations

- Only 1-3 stations per state.
- Urban/Rural is a proxy (`Not in a city`), with only 3 non-city stations.
- No weather data (temperature, wind).

## Team

| Member | Responsibility |
|---|---|
| Rawan | Preprocessing, EDA, feature engineering |
| Alaa | UnSupervised ML and Streamlit app |
| Tasneem | Supervised ML |