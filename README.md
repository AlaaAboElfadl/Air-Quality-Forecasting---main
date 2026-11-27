# Air Quality Forecasting

Pattern analysis and pollution prediction using US EPA air quality data. The project explores historical pollution patterns across US states and cities, analyzes air quality trends, and builds machine learning models to forecast the ozone Air Quality Index (**O3 AQI**).

## Dataset

- **Source:** [US Pollution Data (EPA), Kaggle](https://www.kaggle.com/datasets/sogun3/uspollution/data)
- **Pollutants:** NO2, O3, SO2, CO (daily mean, max value, max hour, AQI)
- **Original Sample:** 70,922 daily records, 37 stations, 29 states, 2010–2015
- **Target:** `O3 AQI`

## Project Structure

```text
Air-Quality-Forecasting/
│
├── app/
│   └── app.py                       # Streamlit dashboard (planned)
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── ml_preprocessing.ipynb
│   ├── supervised_learning.ipynb
│   └── unsupervised_learning.ipynb
│
├── results/
│└── supervised/
│
├── reports/
│   └── unsupervised/
│
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

The full Kaggle dataset is not stored in the repository. The sampled and processed datasets are included.

## Project Workflow

| # | Notebook | Purpose |
|---|---|---|
| 1 | `data_understanding` | Explore the original dataset |
| 2 | `data_cleaning` | Clean missing and invalid values |
| 3 | `EDA` | Analyze pollution trends and patterns |
| 4 | `ml_preprocessing` | Feature engineering and data splitting |
| 5 | `supervised_learning` | Train and evaluate forecasting models |
| 6 | `unsupervised_learning` | Discover pollution patterns using clustering |

## Model-Ready Data (`data/final/`)

| File | Description |
|---|---|
| `X_train`, `X_val`, `X_test` | 25 unscaled features |
| `X_train_std`, `X_val_std`, `X_test_std` | Standardized features |
| `y_train`, `y_val`, `y_test` | Target variable (`O3 AQI`) |

**Chronological Split:**
- Training: 2010–2013
- Validation: 2014
- Testing: 2015

**Features:** Lag values (1, 7, 30 days), rolling statistics, Month, Year, Region, area_type, and Season.

**Preprocessing Notes:**
- Same-day pollutant columns must not be used as input features to avoid data leakage.
- Scalers, encoders, and imputers are fitted on the training set only.

## Project Components

### 1. Data Analysis & Preprocessing — Rawan
- Data cleaning and missing-value handling.
- EDA and pollution pattern analysis.
- Feature engineering for machine learning.

### 2. Supervised Learning — Tasneem
- Regression models for O3 AQI forecasting.
- Model comparison and evaluation using chronological data splits.

### 3. Unsupervised Learning — Alaa
- KMeans for discovering historical pollution patterns.
- Elbow Method and Silhouette Score for cluster selection.
- DBSCAN for identifying dense groups and potential anomalies.

## Results

Project outputs are stored in `results/` and `reports/`, including model comparisons, test predictions, cluster profiles, and DBSCAN results.

## Streamlit Dashboard

A unified Streamlit dashboard is planned to bring all project components together, including:

- **Air Quality Overview**
- **Exploratory Data Analysis**
- **O3 AQI Forecasting**
- **Pollution Pattern Analysis**
- **Model Performance**

## Limitations

- Only 1–3 monitoring stations per state.
- Urban/Rural classification relies on `Not in a city`, with only 3 non-city stations.
- No weather variables are available.
- Clustering identifies pollution behavior patterns, not geographic regions.
- DBSCAN is applied to a sample of the dataset.

