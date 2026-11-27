
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib
import xgboost as xgb


st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🌍",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "df_clean.csv"

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date Local"] = pd.to_datetime(df["Date Local"])
    return df

df = load_data()

# Sidebar
st.sidebar.title("🌍 Air Quality")
st.sidebar.caption("Air Quality Forecasting Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Supervised Learning",
        "Unsupervised Learning",
        "Model Performance",
        "AQI Forecasting"
    ]
)

st.title("Air Quality Forecasting")
st.caption("US EPA Air Quality Analysis & Prediction")


if page == "Overview":

    st.title("🌍 Air Quality Overview")
    st.caption(
        "Explore air pollution patterns across the United States | 2010–2015"
    )

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    total_records = len(df)
    total_stations = df["station"].nunique()
    total_states = df["State"].nunique()
    total_cities = df["City"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Monitoring Stations", total_stations)
    col3.metric("States Covered", total_states)
    col4.metric("Cities Covered", total_cities)

    st.divider()

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.subheader("Explore Air Quality Data")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_state = st.selectbox(
            "Select State",
            ["All States"] + sorted(df["State"].dropna().unique().tolist())
        )

    with filter_col2:
        selected_year = st.selectbox(
            "Select Year",
            ["All Years"] + sorted(df["Year"].unique().tolist())
        )

    filtered_df = df.copy()

    if selected_state != "All States":
        filtered_df = filtered_df[
            filtered_df["State"] == selected_state
        ]

    if selected_year != "All Years":
        filtered_df = filtered_df[
            filtered_df["Year"] == selected_year
        ]

    # -----------------------------
    # POLLUTANT AQI METRICS
    # -----------------------------
    st.subheader("Average Pollutant AQI")

    pollutant_cols = {
        "O3": "O3 AQI",
        "NO2": "NO2 AQI",
        "SO2": "SO2 AQI",
        "CO": "CO AQI"
    }

    metric_cols = st.columns(4)

    for col, (name, column) in zip(metric_cols, pollutant_cols.items()):
        avg_value = filtered_df[column].mean()
        col.metric(
            f"{name} Average AQI",
            f"{avg_value:.2f}" if pd.notna(avg_value) else "N/A"
        )

    st.divider()

    # -----------------------------
    # AQI TREND OVER TIME
    # -----------------------------
    st.subheader("Air Quality Trends Over Time")

    yearly_data = (
        filtered_df.groupby("Year")[
            ["O3 AQI", "NO2 AQI", "SO2 AQI", "CO AQI"]
        ]
        .mean()
        .reset_index()
        .melt(
            id_vars="Year",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_trend = px.line(
        yearly_data,
        x="Year",
        y="Average AQI",
        color="Pollutant",
        markers=True,
        title="Average AQI by Year"
    )

    fig_trend.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        legend_title="Pollutant",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------
    # POLLUTANT COMPARISON
    # -----------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Pollutant Comparison")

        avg_pollutants = (
            filtered_df[list(pollutant_cols.values())]
            .mean()
            .reset_index()
        )

        avg_pollutants.columns = ["Pollutant", "Average AQI"]

        fig_bar = px.bar(
            avg_pollutants,
            x="Pollutant",
            y="Average AQI",
            color="Pollutant",
            title="Average AQI by Pollutant"
        )

        fig_bar.update_layout(
            template="plotly_dark",
            showlegend=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # SEASONAL ANALYSIS
    # -----------------------------
    with chart_col2:
        st.subheader("Seasonal Pollution Analysis")

        seasonal_data = (
            filtered_df.groupby("Season")[
                ["O3 AQI", "NO2 AQI", "SO2 AQI", "CO AQI"]
            ]
            .mean()
            .reset_index()
            .melt(
                id_vars="Season",
                var_name="Pollutant",
                value_name="Average AQI"
            )
        )

        fig_season = px.bar(
            seasonal_data,
            x="Season",
            y="Average AQI",
            color="Pollutant",
            barmode="group",
            title="Average AQI by Season"
        )

        fig_season.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(fig_season, use_container_width=True)

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    with st.expander("View Dataset Preview"):
        st.dataframe(
            filtered_df.head(20),
            use_container_width=True
        )



elif page == "Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")
    st.caption("Explore pollution distributions, trends, correlations and geographic patterns.")

    pollutants = ["O3 AQI", "NO2 AQI", "SO2 AQI", "CO AQI"]

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.subheader("Data Filters")

    c1, c2, c3 = st.columns(3)

    with c1:
        state_options = ["All States"] + sorted(
            df["State"].dropna().unique().tolist()
        )
        selected_state = st.selectbox("State", state_options, key="eda_state")

    with c2:
        year_options = ["All Years"] + sorted(
            df["Year"].dropna().unique().tolist()
        )
        selected_year = st.selectbox("Year", year_options, key="eda_year")

    with c3:
        season_options = ["All Seasons"] + [
            s for s in ["Winter", "Spring", "Summer", "Fall"]
            if s in df["Season"].dropna().unique()
        ]
        selected_season = st.selectbox(
            "Season", season_options, key="eda_season"
        )

    eda_df = df.copy()

    if selected_state != "All States":
        eda_df = eda_df[eda_df["State"] == selected_state]

    if selected_year != "All Years":
        eda_df = eda_df[eda_df["Year"] == selected_year]

    if selected_season != "All Seasons":
        eda_df = eda_df[eda_df["Season"] == selected_season]

    st.caption(f"Records in current selection: {len(eda_df):,}")

    st.divider()

    # -----------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------
    st.header("1. Descriptive Statistics")

    stats = eda_df[pollutants].agg(
        ["mean", "median", "std", "min", "max"]
    ).T.round(2)

    st.dataframe(stats, use_container_width=True)

    # -----------------------------
    # POLLUTANT DISTRIBUTIONS
    # -----------------------------
    st.header("2. Pollutant Distributions")

    selected_pollutant = st.selectbox(
        "Choose Pollutant",
        pollutants,
        key="eda_distribution_pollutant"
    )

    fig_hist = px.histogram(
        eda_df,
        x=selected_pollutant,
        nbins=40,
        marginal="box",
        title=f"{selected_pollutant} Distribution",
        color_discrete_sequence=["#3498DB"]
    )

    fig_hist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------
    # TEMPORAL ANALYSIS
    # -----------------------------
    st.header("3. Temporal Analysis")

    monthly = (
        eda_df.groupby("Month")[pollutants]
        .mean()
        .reset_index()
        .melt(
            id_vars="Month",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_month = px.line(
        monthly,
        x="Month",
        y="Average AQI",
        color="Pollutant",
        markers=True,
        title="Monthly Average AQI"
    )

    fig_month.update_layout(
        template="plotly_dark",
        xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig_month, use_container_width=True)

    yearly = (
        eda_df.groupby("Year")[pollutants]
        .mean()
        .reset_index()
        .melt(
            id_vars="Year",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_year = px.line(
        yearly,
        x="Year",
        y="Average AQI",
        color="Pollutant",
        markers=True,
        title="Yearly Average AQI"
    )

    fig_year.update_layout(template="plotly_dark")
    st.plotly_chart(fig_year, use_container_width=True)

    # -----------------------------
    # GEOGRAPHIC ANALYSIS
    # -----------------------------
    st.header("4. Geographic Analysis")

    state_avg = (
        eda_df.groupby("State")[pollutants]
        .mean()
        .reset_index()
        .sort_values("O3 AQI", ascending=False)
    )

    fig_state = px.bar(
        state_avg,
        x="State",
        y="O3 AQI",
        color="O3 AQI",
        title="Average O3 AQI by State",
        color_continuous_scale="Blues"
    )

    fig_state.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_state, use_container_width=True)

    region_avg = (
        eda_df.groupby("Region")[pollutants]
        .mean()
        .reset_index()
        .melt(
            id_vars="Region",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_region = px.bar(
        region_avg,
        x="Region",
        y="Average AQI",
        color="Pollutant",
        barmode="group",
        title="Pollution Comparison by Region"
    )

    fig_region.update_layout(template="plotly_dark")
    st.plotly_chart(fig_region, use_container_width=True)

    # -----------------------------
    # CORRELATION ANALYSIS
    # -----------------------------
    st.header("5. Correlation Analysis")

    corr_method = st.radio(
        "Correlation Method",
        ["Pearson", "Spearman"],
        horizontal=True
    )

    method = corr_method.lower()
    correlation = eda_df[pollutants].corr(method=method)

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"{corr_method} Correlation Between Pollutants"
    )

    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------------
    # SEASONAL ANALYSIS
    # -----------------------------
    st.header("6. Seasonal Analysis")

    season_order = ["Winter", "Spring", "Summer", "Fall"]

    seasonal = (
        eda_df.groupby("Season")[pollutants]
        .mean()
        .reindex(season_order)
        .dropna(how="all")
        .reset_index()
        .melt(
            id_vars="Season",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_season = px.bar(
        seasonal,
        x="Season",
        y="Average AQI",
        color="Pollutant",
        barmode="group",
        category_orders={"Season": season_order},
        title="Average AQI Across Seasons"
    )

    fig_season.update_layout(template="plotly_dark")
    st.plotly_chart(fig_season, use_container_width=True)

    # -----------------------------
    # URBAN VS NON-CITY
    # -----------------------------
    st.header("7. Urban vs Non-city Comparison")

    eda_df["Area Type"] = eda_df["City"].apply(
        lambda city: "Rural/Non-city"
        if city == "Not in a city"
        else "Urban"
    )

    area_summary = (
        eda_df.groupby("Area Type")[pollutants]
        .mean()
        .reset_index()
        .melt(
            id_vars="Area Type",
            var_name="Pollutant",
            value_name="Average AQI"
        )
    )

    fig_area = px.bar(
        area_summary,
        x="Pollutant",
        y="Average AQI",
        color="Area Type",
        barmode="group",
        title="Urban vs Non-city Average AQI"
    )

    fig_area.update_layout(template="plotly_dark")
    st.plotly_chart(fig_area, use_container_width=True)

    st.warning(
        "Non-city classification is only a rural proxy. "
        "The dataset contains just 3 non-city monitoring stations."
    )

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    with st.expander("View Filtered Dataset"):
        st.dataframe(eda_df, use_container_width=True)



elif page == "Supervised Learning":

    st.title("🤖 Supervised Learning")
    st.caption(
        "O3 AQI Prediction using Machine Learning Regression Models"
    )

    # -----------------------------
    # LOAD MODEL RESULTS
    # -----------------------------
    COMPARISON_PATH = BASE_DIR / "results" / "final_model_comparison.csv"
    TEST_RESULTS_PATH = BASE_DIR / "results" / "final_test_results_2015.csv"

    if not COMPARISON_PATH.exists() or not TEST_RESULTS_PATH.exists():
        st.error("Model results files were not found in the results folder.")
        st.stop()

    comparison_df = pd.read_csv(COMPARISON_PATH)
    test_df = pd.read_csv(TEST_RESULTS_PATH)

    # -----------------------------
    # TEST PERFORMANCE METRICS
    # -----------------------------
    st.header("1. Test Performance (2015)")

    best_rmse = comparison_df.loc[
        comparison_df["Test RMSE (2015)"].idxmin()
    ]

    best_mae = comparison_df.loc[
        comparison_df["Test MAE (2015)"].idxmin()
    ]

    best_r2 = comparison_df.loc[
        comparison_df["Test R2 (2015)"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Test RMSE",
        f"{best_rmse['Test RMSE (2015)']:.3f}",
        best_rmse["Model"]
    )

    col2.metric(
        "Best Test MAE",
        f"{best_mae['Test MAE (2015)']:.3f}",
        best_mae["Model"]
    )

    col3.metric(
        "Best Test R²",
        f"{best_r2['Test R2 (2015)']:.3f}",
        best_r2["Model"]
    )

    st.divider()

    # -----------------------------
    # MODEL COMPARISON TABLE
    # -----------------------------
    st.header("2. Model Comparison")

    display_df = comparison_df[
        [
            "Model",
            "CV Strategy",
            "CV RMSE",
            "CV MAE",
            "CV R2",
            "Test RMSE (2015)",
            "Test MAE (2015)",
            "Test R2 (2015)"
        ]
    ].copy()

    st.dataframe(
        display_df.style.format({
            "CV RMSE": "{:.3f}",
            "CV MAE": "{:.3f}",
            "CV R2": "{:.3f}",
            "Test RMSE (2015)": "{:.3f}",
            "Test MAE (2015)": "{:.3f}",
            "Test R2 (2015)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------
    # RMSE COMPARISON
    # -----------------------------
    st.header("3. Error Metrics Comparison")

    rmse_df = comparison_df[
        ["Model", "CV RMSE", "Test RMSE (2015)"]
    ].melt(
        id_vars="Model",
        var_name="Evaluation",
        value_name="RMSE"
    )

    fig_rmse = px.bar(
        rmse_df,
        x="Model",
        y="RMSE",
        color="Evaluation",
        barmode="group",
        title="RMSE: Cross-Validation vs Test (2015)"
    )

    fig_rmse.update_layout(template="plotly_dark")
    st.plotly_chart(fig_rmse, use_container_width=True)

    # -----------------------------
    # MAE AND R2 COMPARISON
    # -----------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        mae_df = comparison_df[
            ["Model", "CV MAE", "Test MAE (2015)"]
        ].melt(
            id_vars="Model",
            var_name="Evaluation",
            value_name="MAE"
        )

        fig_mae = px.bar(
            mae_df,
            x="Model",
            y="MAE",
            color="Evaluation",
            barmode="group",
            title="MAE Comparison"
        )

        fig_mae.update_layout(template="plotly_dark")
        st.plotly_chart(fig_mae, use_container_width=True)

    with chart_col2:

        r2_df = comparison_df[
            ["Model", "CV R2", "Test R2 (2015)"]
        ].melt(
            id_vars="Model",
            var_name="Evaluation",
            value_name="R2"
        )

        fig_r2 = px.bar(
            r2_df,
            x="Model",
            y="R2",
            color="Evaluation",
            barmode="group",
            title="R² Score Comparison"
        )

        fig_r2.update_layout(template="plotly_dark")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    # -----------------------------
    # FINAL TEST RESULTS
    # -----------------------------
    st.header("4. Final Test Results — 2015")

    st.dataframe(
        test_df.style.format({
            "RMSE": "{:.3f}",
            "MAE": "{:.3f}",
            "R2": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # INTERPRETATION
    # -----------------------------
    st.header("5. Model Evaluation Summary")

    st.write(
        f"""
        - **Best RMSE:** {best_rmse['Model']}
        - **Best MAE:** {best_mae['Model']}
        - **Best R²:** {best_r2['Model']}

        The three models achieved relatively close performance
        on the 2015 test dataset.
        """
    )

    st.divider()

    # -----------------------------
    # FEATURE IMPORTANCE ANALYSIS
    # -----------------------------
    st.header("6. Feature Importance Analysis")

    FEATURE_IMPORTANCE_PATH = (
        BASE_DIR / "results" / "feature_importance.csv"
    )

    if FEATURE_IMPORTANCE_PATH.exists():

        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

        selected_model = st.selectbox(
            "Select Model for Feature Importance",
            importance_df["Model"].unique()
        )

        model_importance = importance_df[
            importance_df["Model"] == selected_model
        ].copy()

        model_importance = model_importance.sort_values(
            "Importance",
            ascending=False
        )

        top_feature = model_importance.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "Most Important Feature",
            top_feature["Feature"]
        )

        col2.metric(
            "Importance Score",
            f"{top_feature['Importance']:.2%}"
        )

        top_10 = model_importance.head(10).sort_values(
            "Importance",
            ascending=True
        )

        fig_importance = px.bar(
            top_10,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 10 Features - {selected_model}",
            text=top_10["Importance"].map(
                lambda x: f"{x:.1%}"
            )
        )

        fig_importance.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )

        fig_importance.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

        st.subheader("Feature Importance Ranking")

        display_importance = model_importance.copy()

        display_importance["Importance (%)"] = (
            display_importance["Importance"] * 100
        ).round(2)

        display_importance.insert(
            0,
            "Rank",
            range(1, len(display_importance) + 1)
        )

        st.dataframe(
            display_importance[
                ["Rank", "Feature", "Importance (%)"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"The most influential feature for {selected_model} "
            f"is {top_feature['Feature']}, with an importance "
            f"score of {top_feature['Importance']:.2%}."
        )

        st.caption(
            "Feature importance shows the relative contribution "
            "of each feature to the model. It does not indicate "
            "whether the feature increases or decreases AQI."
        )

    else:
        st.warning("Feature importance file was not found.")




elif page == "Unsupervised Learning":

    st.title("🔍 Unsupervised Learning")
    st.caption(
        "Discovering hidden pollution patterns using KMeans "
        "Clustering and DBSCAN Anomaly Detection."
    )

    # --------------------------------
    # LOAD UNSUPERVISED RESULTS
    # --------------------------------
    UNSUPERVISED_DIR = BASE_DIR / "reports" / "unsupervised"

    required_files = {
        "distribution": "kmeans_cluster_distribution.csv",
        "cluster_profile": "kmeans_cluster_profile.csv",
        "pollutant_profile": "kmeans_pollutant_profile.csv",
        "k_selection": "kmeans_k_selection_metrics.csv",
        "dbscan": "dbscan_sample_results.csv",
    }

    missing_files = [
        filename
        for filename in required_files.values()
        if not (UNSUPERVISED_DIR / filename).exists()
    ]

    if missing_files:
        st.error(
            "Some Unsupervised Learning result files are missing:"
        )
        for filename in missing_files:
            st.write(f"- {filename}")
        st.stop()

    distribution_df = pd.read_csv(
        UNSUPERVISED_DIR / required_files["distribution"],
        index_col=0
    )

    cluster_profile_df = pd.read_csv(
        UNSUPERVISED_DIR / required_files["cluster_profile"],
        index_col=0
    )

    pollutant_profile_df = pd.read_csv(
        UNSUPERVISED_DIR / required_files["pollutant_profile"],
        index_col=0
    )

    k_selection_df = pd.read_csv(
        UNSUPERVISED_DIR / required_files["k_selection"]
    )

    dbscan_df = pd.read_csv(
        UNSUPERVISED_DIR / required_files["dbscan"]
    )

    # --------------------------------
    # 1. CLUSTERING OVERVIEW
    # --------------------------------
    st.header("1. Clustering Overview")

    total_records = int(
        distribution_df["Number of records"].sum()
    )

    number_of_clusters = len(distribution_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Number of Clusters",
        number_of_clusters
    )

    col3.metric(
        "Selected K",
        number_of_clusters
    )

    st.info(
        """
        KMeans was used to group observations based on historical
        pollutant AQI patterns. The clustering uses lag and rolling
        features for O3, NO2, SO2 and CO.
        """
    )

    st.divider()

    # --------------------------------
    # 2. KMEANS CLUSTER DISTRIBUTION
    # --------------------------------
    st.header("2. KMeans Cluster Distribution")

    st.write(
        "Distribution of records across the identified clusters."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_distribution = px.bar(
            distribution_df.reset_index(),
            x=distribution_df.index.name or "index",
            y="Number of records",
            title="Number of Records per Cluster",
            text="Number of records"
        )

        fig_distribution.update_layout(
            template="plotly_dark",
            xaxis_title="Cluster",
            yaxis_title="Number of Records",
            height=400
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )

    with col2:

        fig_pie = px.pie(
            distribution_df.reset_index(),
            names=distribution_df.index.name or "index",
            values="Number of records",
            title="Cluster Percentage Distribution",
            hole=0.4
        )

        fig_pie.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    st.dataframe(
        distribution_df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------
    # 3. KMEANS CLUSTER PROFILES
    # --------------------------------
    st.header("3. Cluster Profiles")

    st.write(
        """
        This section compares the average historical pollution
        features across the KMeans clusters.
        """
    )

    fig_cluster_profile = px.imshow(
        cluster_profile_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="KMeans Cluster Profile - Historical Features"
    )

    fig_cluster_profile.update_layout(
        template="plotly_dark",
        height=550
    )

    st.plotly_chart(
        fig_cluster_profile,
        use_container_width=True
    )

    st.caption(
        """
        Positive values indicate above-reference feature values,
        while negative values indicate below-reference values.
        These are scaled relative profiles, not absolute AQI levels.
        """
    )

    st.divider()

    # --------------------------------
    # 4. POLLUTANT PROFILE
    # --------------------------------
    st.header("4. Pollutant Profile Analysis")

    st.write(
        "Average relative pollution patterns for each pollutant."
    )

    fig_pollutant = px.imshow(
        pollutant_profile_df,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Relative Pollutant Profile by Cluster"
    )

    fig_pollutant.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig_pollutant,
        use_container_width=True
    )

    st.divider()

    # --------------------------------
    # 5. SELECTING THE OPTIMAL K
    # --------------------------------
    st.header("5. Selecting the Optimal K")

    st.write(
        """
        We evaluated different K values using the Elbow Method
        and Silhouette Score to understand the clustering quality.
        """
    )

    if "K" in k_selection_df.columns:

        metric_cols = st.columns(2)

        with metric_cols[0]:

            if "Inertia" in k_selection_df.columns:

                fig_elbow = px.line(
                    k_selection_df,
                    x="K",
                    y="Inertia",
                    markers=True,
                    title="Elbow Method"
                )

                fig_elbow.update_layout(
                    template="plotly_dark",
                    height=350
                )

                st.plotly_chart(
                    fig_elbow,
                    use_container_width=True
                )

        with metric_cols[1]:

            if "Silhouette Score" in k_selection_df.columns:

                fig_silhouette = px.line(
                    k_selection_df,
                    x="K",
                    y="Silhouette Score",
                    markers=True,
                    title="Silhouette Score"
                )

                fig_silhouette.update_layout(
                    template="plotly_dark",
                    height=350
                )

                st.plotly_chart(
                    fig_silhouette,
                    use_container_width=True
                )


    # --------------------------------
    # CLUSTER INTERPRETATION
    # --------------------------------
    st.header("6. Understanding the Clusters")

    st.write(
        "The following interpretation explains the relative "
        "pollution patterns identified by KMeans."
    )

    cluster_0 = pollutant_profile_df.loc[0]
    cluster_1 = pollutant_profile_df.loc[1]

    pollutants = ["O3", "NO2", "SO2", "CO"]

    cluster_col1, cluster_col2 = st.columns(2)

    with cluster_col1:

        st.subheader("🟠 Cluster 0")

        for pollutant in pollutants:

            value = cluster_0[pollutant]

            if value > 0:
                description = "Above average"
            elif value < 0:
                description = "Below average"
            else:
                description = "Around average"

            st.write(
                f"**{pollutant}:** {description} ({value:.3f})"
            )

        st.info(
            "Cluster 0 is characterized by relatively lower O3 "
            "and higher NO2, SO2 and CO levels."
        )

    with cluster_col2:

        st.subheader("🔵 Cluster 1")

        for pollutant in pollutants:

            value = cluster_1[pollutant]

            if value > 0:
                description = "Above average"
            elif value < 0:
                description = "Below average"
            else:
                description = "Around average"

            st.write(
                f"**{pollutant}:** {description} ({value:.3f})"
            )

        st.info(
            "Cluster 1 is characterized by relatively higher O3 "
            "and lower NO2, SO2 and CO levels."
        )

    st.caption(
        "Interpretation is based on scaled relative profiles. "
        "Above average does not necessarily mean unhealthy AQI."
    )

    st.divider()

    # --------------------------------
    # 7. DBSCAN ANOMALY DETECTION
    # --------------------------------
    st.header("7. DBSCAN Anomaly Detection")

    st.write(
        """
        DBSCAN identifies dense groups of observations and
        labels observations outside these groups as noise.
        """
    )

    if "DBSCAN_Cluster" in dbscan_df.columns:

        dbscan_counts = (
            dbscan_df["DBSCAN_Cluster"]
            .value_counts()
            .sort_index()
        )

        noise_count = int(
            (dbscan_df["DBSCAN_Cluster"] == -1).sum()
        )

        sample_size = len(dbscan_df)

        noise_percentage = (
            noise_count / sample_size * 100
            if sample_size > 0 else 0
        )

        dense_clusters = (
            dbscan_df.loc[
                dbscan_df["DBSCAN_Cluster"] != -1,
                "DBSCAN_Cluster"
            ].nunique()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "DBSCAN Sample Size",
            f"{sample_size:,}"
        )

        col2.metric(
            "Dense Clusters",
            dense_clusters
        )

        col3.metric(
            "Noise Percentage",
            f"{noise_percentage:.2f}%"
        )

        dbscan_chart_df = dbscan_counts.rename(
            "Number of Records"
        ).reset_index()

        dbscan_chart_df["DBSCAN_Cluster"] = (
            dbscan_chart_df["DBSCAN_Cluster"].astype(str)
        )

        dbscan_chart_df["Type"] = dbscan_chart_df[
            "DBSCAN_Cluster"
        ].apply(
            lambda x: "Noise" if x == "-1" else "Cluster"
        )

        fig_dbscan = px.bar(
            dbscan_chart_df,
            x="DBSCAN_Cluster",
            y="Number of Records",
            color="Type",
            title="DBSCAN Cluster Distribution",
            text="Number of Records"
        )

        fig_dbscan.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="DBSCAN Label (-1 = Noise)",
            yaxis_title="Number of Records"
        )

        st.plotly_chart(
            fig_dbscan,
            use_container_width=True
        )

        st.warning(
            """
            DBSCAN noise points are potential anomalies in the
            feature space. They are not automatically incorrect
            measurements or harmful pollution events.
            """
        )

    else:
        st.warning(
            "DBSCAN_Cluster column was not found in the results file."
        )

    st.divider()

    # --------------------------------
    # 8. DBSCAN DETAILED ANALYSIS
    # --------------------------------
    st.header("8. DBSCAN Detailed Analysis")

    if "DBSCAN_Cluster" in dbscan_df.columns:

        dbscan_analysis_df = dbscan_df.copy()
        dbscan_analysis_df["Point_Type"] = dbscan_analysis_df[
            "DBSCAN_Cluster"
        ].apply(lambda value: "Noise" if value == -1 else "Normal")

        # Noise distribution across chronological data splits
        if "data_split" in dbscan_analysis_df.columns:

            st.subheader("Noise Distribution Across Data Splits")

            split_summary = pd.crosstab(
                dbscan_analysis_df["data_split"],
                dbscan_analysis_df["Point_Type"]
            )

            for category in ["Normal", "Noise"]:
                if category not in split_summary.columns:
                    split_summary[category] = 0

            split_summary["Noise Rate (%)"] = (
                split_summary["Noise"]
                / (split_summary["Noise"] + split_summary["Normal"])
                * 100
            ).round(2)

            split_summary = split_summary.reset_index()

            split_col1, split_col2 = st.columns(2)

            with split_col1:
                fig_split_counts = px.bar(
                    split_summary,
                    x="data_split",
                    y=["Normal", "Noise"],
                    barmode="group",
                    title="Normal vs Noise Points by Data Split"
                )
                fig_split_counts.update_layout(template="plotly_dark")
                st.plotly_chart(
                    fig_split_counts,
                    use_container_width=True
                )

            with split_col2:
                fig_split_rate = px.bar(
                    split_summary,
                    x="data_split",
                    y="Noise Rate (%)",
                    title="Noise Rate by Data Split"
                )
                fig_split_rate.update_layout(template="plotly_dark")
                st.plotly_chart(
                    fig_split_rate,
                    use_container_width=True
                )

            st.dataframe(
                split_summary,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # Compare average lag-1 pollutant features
        st.subheader("Noise vs Normal: Pollution Patterns")

        comparison_features = [
            "O3_AQI_lag1",
            "NO2_AQI_lag1",
            "SO2_AQI_lag1",
            "CO_AQI_lag1"
        ]

        available_features = [
            feature for feature in comparison_features
            if feature in dbscan_analysis_df.columns
        ]

        if available_features:
            comparison_df = (
                dbscan_analysis_df.groupby("Point_Type")[available_features]
                .mean()
                .reindex(["Normal", "Noise"])
                .T
                .reset_index()
            )

            comparison_df.columns = [
                "Pollutant Feature", "Normal", "Noise"
            ]

            fig_comparison = px.bar(
                comparison_df,
                x="Pollutant Feature",
                y=["Normal", "Noise"],
                barmode="group",
                title="Average Lag-1 Pollution Features: Noise vs Normal"
            )
            fig_comparison.update_layout(template="plotly_dark")
            st.plotly_chart(
                fig_comparison,
                use_container_width=True
            )
        else:
            st.info(
                "Lag-1 pollutant features are not available in the DBSCAN results."
            )

        st.info(
            "Noise points are observations that DBSCAN could not assign "
            "to a dense cluster. They are not necessarily incorrect data "
            "or the most polluted observations."
        )

    st.divider()

    # --------------------------------
    # 9. METHODOLOGY AND LIMITATIONS
    # --------------------------------
    st.header("9. Methodology & Limitations")

    with st.expander("KMeans Methodology"):

        st.write(
            """
            - KMeans groups observations based on historical AQI patterns.
            - The selected solution uses K = 2.
            - Clustering features include lag and rolling statistics
              for O3, NO2, SO2 and CO.
            - Location and encoded categorical columns are excluded
              from clustering distances.
            """
        )

    with st.expander("DBSCAN Methodology"):

        st.write(
            """
            - DBSCAN was applied to a reproducible sample of up to
              8,000 observations.
            - min_samples = 10.
            - eps = 3.0 in the saved analysis.
            - Label -1 represents noise points.
            """
        )

    with st.expander("Important Limitations"):

        st.write(
            """
            - Clusters represent recurring pollution patterns,
              not geographic regions.
            - Clusters are not formal AQI health categories.
            - DBSCAN results are sample-based, not full-dataset results.
            - Noise points require investigation and should not
              automatically be considered data errors.
            """
        )


elif page == "Model Performance":
    st.header("Model Performance")
    st.info("Model comparison and evaluation metrics will be added here.")


elif page == "AQI Forecasting":

    st.header("🌍 O3 AQI Forecasting")
    st.caption("Predict O3 AQI using trained machine learning models.")

    MODELS_PATH = BASE_DIR / "models" / "streamlit_artifacts" / "supervised_models.joblib"
    X_TEST_PATH = BASE_DIR / "data" / "final" / "X_test.csv"
    Y_TEST_PATH = BASE_DIR / "data" / "final" / "y_test.csv"

    @st.cache_resource
    def load_supervised_models():
        return joblib.load(MODELS_PATH)

    @st.cache_data
    def load_test_data():
        X = pd.read_csv(X_TEST_PATH)
        y = pd.read_csv(Y_TEST_PATH)
        return X, y

    models = load_supervised_models()
    X_test, y_test = load_test_data()

    st.subheader("1. Select Test Sample")

    sample_index = st.slider(
        "Select sample",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0
    )

    selected_model = st.selectbox(
        "Choose Prediction Model",
        list(models.keys())
    )

    actual_value = float(y_test.iloc[sample_index, 0])

    if st.button("🚀 Predict O3 AQI", use_container_width=True):

        predictions = {}

        for model_name, model in models.items():

            features = list(model.feature_names_in_)
            input_data = X_test.iloc[[sample_index]][features]

            if model_name == "XGBoost":
                dmatrix = xgb.DMatrix(
                    input_data,
                    feature_names=features
                )
                prediction = model.get_booster().predict(dmatrix)[0]

            else:
                prediction = model.predict(input_data)[0]

            predictions[model_name] = float(prediction)

        selected_prediction = predictions[selected_model]
        error = abs(actual_value - selected_prediction)

        st.divider()
        st.subheader("2. Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted O3 AQI",
            f"{selected_prediction:.2f}"
        )

        col2.metric(
            "Actual O3 AQI",
            f"{actual_value:.2f}"
        )

        col3.metric(
            "Absolute Error",
            f"{error:.2f}"
        )

        st.success(f"Prediction completed using {selected_model}.")

        st.divider()
        st.subheader("3. Compare All Models")

        comparison = pd.DataFrame({
            "Model": list(predictions.keys()),
            "Predicted O3 AQI": list(predictions.values())
        })

        comparison["Actual O3 AQI"] = actual_value
        comparison["Absolute Error"] = (
            comparison["Predicted O3 AQI"] - actual_value
        ).abs()

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="Predicted O3 AQI",
            color="Model",
            title="O3 AQI Predictions by Model"
        )

        fig.add_hline(
            y=actual_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Actual O3 AQI"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)
