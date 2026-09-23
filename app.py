import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bantay Dengue: Data-Driven Monitoring Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem; font-weight: bold; color: #1B5E20;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #1B5E20;
    }
    .sub-header {
        font-size: 1.05rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B5E20 !important; color: white !important;
    }
    .warning-box {
        background-color: #FFF3E0; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #FF9800; margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #2196F3; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=True)
def load_dengue_data():
    """Load the PHO-ADS dengue surveillance dataset."""
    search_paths = ['.', '/content/data', '/content']
    possible_files = [
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.xlsx',
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.xls',
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.csv',
        'dengue_data.xlsx',
        'dengue_data.csv'
    ]
    
    filepath = None
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        for f in possible_files:
            full_path = os.path.join(path, f)
            if os.path.exists(full_path):
                filepath = full_path
                break
        if filepath:
            break
    
    # Also search for any file containing "dengue" or "FINAL-ADS"
    if filepath is None:
        for path in search_paths:
            if not os.path.isdir(path):
                continue
            for ext in ['*.xlsx', '*.xls', '*.csv']:
                for f in glob.glob(os.path.join(path, ext)):
                    fname = os.path.basename(f).lower()
                    if 'dengue' in fname or 'final-ads' in fname or 'bantay' in fname:
                        filepath = f
                        break
                if filepath:
                    break
            if filepath:
                break
    
    if filepath is None:
        return None, "No dengue dataset file found."
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        return df, f"Loaded: {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Error loading file: {e}"


# ============================================================
# DATA PREPROCESSING (KDD PROCESS - PHASE 1 & 2)
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess_data(df):
    """
    Chapter 3 - Preprocessing of Data (KDD Process)
    - Data Cleaning: identify duplicates, handle missing values
    - Data Transformation: standardize categorical variables
    - Data Standardization: geographic standardization, temporal harmonization
    """
    df = df.copy()
    
    # ---- Phase 1: Data Cleaning ----
    # Remove header rows that may have been duplicated
    df = df[df['Year'].astype(str).str.match(r'^\d{4}$', na=False)].copy()
    
    # Convert Year to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Convert Morbidity Week to numeric
    df['MORBIDITY WEEK'] = pd.to_numeric(df['MORBIDITY WEEK'], errors='coerce')
    
    # Convert Age to numeric
    df['AgeYears'] = pd.to_numeric(df['AgeYears'], errors='coerce')
    
    # Convert dates
    date_cols = ['Date Consulted', 'DAdmit', 'DOnset', 'DateDied']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing key values
    df = df.dropna(subset=['Year', 'MORBIDITY WEEK', 'AgeYears'])
    
    # ---- Phase 2: Data Transformation ----
    # Standardize Sex
    df['Sex'] = df['Sex'].astype(str).str.strip().str.capitalize()
    df['Sex'] = df['Sex'].replace({'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'})
    
    # Standardize Municipality names
    municipality_col = '(Permanent Address) City / Municipality'
    if municipality_col in df.columns:
        df['Municipality'] = df[municipality_col].astype(str).str.strip().str.title()
    else:
        df['Municipality'] = 'Unknown'
    
    # Standardize Barangay
    barangay_col = '(Permanent Address) Barangay'
    if barangay_col in df.columns:
        df['Barangay'] = df[barangay_col].astype(str).str.strip().str.title()
    else:
        df['Barangay'] = 'Unknown'
    
    # Standardize Clinical Classification
    if 'ClinClass' in df.columns:
        df['Clinical_Classification'] = df['ClinClass'].astype(str).str.strip()
        df['Clinical_Classification'] = df['Clinical_Classification'].replace({
            'Dengue Without Warning Signs': 'Dengue Without Warning Signs',
            'Dengue With Warning Signs': 'Dengue With Warning Signs',
            'Severe Dengue': 'Severe Dengue'
        })
    else:
        df['Clinical_Classification'] = 'Unknown'
    
    # Standardize Case Classification
    if 'Case Classification' in df.columns:
        df['Case_Classification'] = df['Case Classification'].astype(str).str.strip().str.title()
    else:
        df['Case_Classification'] = 'Unknown'
    
    # Standardize Outcome
    if 'OUTCOME' in df.columns:
        df['Outcome'] = df['OUTCOME'].astype(str).str.strip().str.title()
    else:
        df['Outcome'] = 'Unknown'
    
    # Create Age Groups (WHO 2009 dengue surveillance reporting)
    bins = [0, 4, 9, 17, 59, 150]
    labels = ['0-4', '5-9', '10-17', '18-59', '60+']
    df['Age_Group'] = pd.cut(df['AgeYears'], bins=bins, labels=labels, right=True)
    
    # Create temporal features
    df['Date_Onset'] = pd.to_datetime(df['DOnset'], errors='coerce')
    df['Month'] = df['Date_Onset'].dt.month
    df['Quarter'] = df['Date_Onset'].dt.quarter
    df['Week'] = df['MORBIDITY WEEK']
    
    # Season classification (Philippines: Dry = Dec-May, Wet = Jun-Nov)
    def get_season(month):
        if pd.isna(month):
            return 'Unknown'
        month = int(month)
        if month in [12, 1, 2, 3, 4, 5]:
            return 'Dry Season'
        else:
            return 'Wet Season'
    df['Season'] = df['Month'].apply(get_season)
    
    # Create severity indicator
    df['Severity'] = df['Clinical_Classification'].apply(
        lambda x: 'High' if 'Severe' in str(x) else ('Medium' if 'Warning' in str(x) else 'Low')
    )
    
    return df


# ============================================================
# LOAD & PREPROCESS
# ============================================================
df_raw, load_message = load_dengue_data()

st.markdown('<div class="main-header">🦟 Bantay Dengue: A Data-Driven Monitoring Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Data-Driven Monitoring Approach for Dengue Incidence in Agusan del Sur<br>'
            '<i>Provincial Health Office - Agusan del Sur (PHO-ADS) • 2021–2026</i></div>',
            unsafe_allow_html=True)

if df_raw is None:
    st.error(f"❌ {load_message}")
    st.info("Please ensure the dengue dataset file is in the correct location. "
            "Expected file: `FINAL-ADS-DENGUE-DATABASE-2021-2025.xlsx`")
    st.stop()
else:
    st.success(f"✅ {load_message}")

df = preprocess_data(df_raw)

# Display dataset summary
with st.expander("📊 Dataset Summary (PHO-ADS Dengue Surveillance Data)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", f"{len(df):,}")
    c2.metric("Municipalities", df['Municipality'].nunique())
    c3.metric("Barangays", df['Barangay'].nunique())
    c4.metric("Years Covered", f"{int(df['Year'].min())}–{int(df['Year'].max())}")
    
    st.markdown("**Cases per Year:**")
    yearly = df['Year'].value_counts().sort_index()
    st.dataframe(yearly.rename('Cases').to_frame(), use_container_width=True)
    
    st.markdown("**Clinical Classification Distribution:**")
    clin = df['Clinical_Classification'].value_counts()
    st.dataframe(clin.rename('Count').to_frame().assign(
        Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
    ), use_container_width=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Dashboard Controls")

municipality_filter = st.sidebar.multiselect(
    "Filter by Municipality:",
    options=sorted(df['Municipality'].dropna().unique()),
    default=sorted(df['Municipality'].dropna().unique())[:5]
)

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

k_clusters = st.sidebar.slider("K-Means: Number of Clusters", 2, 8, 4)
arima_forecast_steps = st.sidebar.slider("ARIMA: Forecast Horizon (months)", 3, 12, 6)
rf_estimators = st.sidebar.slider("Random Forest: Number of Trees", 50, 300, 100, 50)

# Filter data
df_f = df[
    (df['Municipality'].isin(municipality_filter)) &
    (df['Year'].between(year_range[0], year_range[1]))
].copy()

if len(df_f) < 10:
    st.warning("⚠️ Too few records after filtering. Please adjust the sidebar filters.")
    st.stop()


# ============================================================
# TABS (Based on Chapter 1 Objectives & Chapter 3 Methodology)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Temporal Analysis (ARIMA)",
    "🗺️ Geographic Clustering (K-Means)",
    "👥 Demographic Profiling",
    "🔬 Classification (Random Forest)",
    "💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Dengue Cases</p><h2>{len(df_f):,}</h2></div>', 
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Municipalities</p><h2>{df_f["Municipality"].nunique()}</h2></div>', 
                unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>', 
                unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Severe Cases</p><h2>{(df_f["Severity"]=="High").sum():,}</h2></div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📌 Dengue Incidence by Municipality")
    mun_counts = df_f['Municipality'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    mun_counts.plot(kind='bar', color='#1B5E20', edgecolor='black', ax=ax)
    ax.set_ylabel("Number of Cases")
    ax.set_title("Dengue Cases per Municipality")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("📋 Clinical Classification Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        clin_counts = df_f['Clinical_Classification'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#FFA726', '#EF5350', '#B71C1C']
        ax.pie(clin_counts.values, labels=clin_counts.index, autopct='%1.1f%%',
               colors=colors[:len(clin_counts)], startangle=90)
        ax.set_title("Clinical Classification Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(clin_counts.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)
    
    st.subheader("⚠️ Case Outcomes")
    outcome_counts = df_f['Outcome'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    outcome_counts.plot(kind='bar', color=['#2E7D32', '#C62828'], edgecolor='black', ax=ax)
    ax.set_ylabel("Number of Cases")
    ax.set_title("Case Outcomes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ------------------------------------------------------------
# TAB 2: TEMPORAL ANALYSIS (ARIMA)
# ------------------------------------------------------------
with tab2:
    st.header("📈 Time Series Analysis (ARIMA)")
    st.caption("Chapter 3: Time Series Analysis (ARIMA) — Identifying seasonal patterns and forecasting future outbreaks.")
    
    # Aggregate monthly data
    ts_df = df_f.dropna(subset=['Date_Onset']).copy()
    ts_df['ym'] = ts_df['Date_Onset'].dt.to_period('M').dt.to_timestamp()
    monthly = ts_df.groupby('ym').size().rename('cases').asfreq('MS').fillna(0)
    
    if len(monthly) < 12:
        st.warning(f"Not enough monthly data ({len(monthly)} months) to fit ARIMA. "
                   "Broaden the year range or municipality filter (need at least 12 months).")
    else:
        # Historical plot
        st.markdown("### 📊 Historical Monthly Dengue Cases")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly.index, monthly.values, marker='o', color='#1B5E20', markersize=4)
        ax.set_ylabel("Number of Cases")
        ax.set_title("Monthly Dengue Incidence Trend")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Stationarity test
        st.markdown("---")
        st.markdown("### 🔬 Stationarity Test (Augmented Dickey-Fuller)")
        try:
            adf_result = adfuller(monthly.dropna(), autolag='AIC')
            adf_stat, adf_p = adf_result[0], adf_result[1]
            c1, c2, c3 = st.columns(3)
            c1.metric("ADF Statistic", f"{adf_stat:.4f}")
            c2.metric("p-value", f"{adf_p:.4f}")
            c3.metric("Conclusion", "Stationary" if adf_p < 0.05 else "Non-stationary")
            
            if adf_p < 0.05:
                st.success("✅ Series is stationary (p < 0.05). ARIMA with d=0 may work.")
                suggested_d = 0
            else:
                st.info("ℹ️ Series is non-stationary (p ≥ 0.05). Differencing (d=1) recommended.")
                suggested_d = 1
        except Exception as e:
            st.warning(f"ADF test failed: {e}")
            suggested_d = 1
        
        # ACF/PACF
        st.markdown("---")
        st.markdown("### 📈 ACF and PACF Plots")
        col1, col2 = st.columns(2)
        try:
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3))
                plot_acf(monthly.dropna(), lags=min(20, len(monthly)-1), ax=ax)
                ax.set_title("ACF")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3))
                plot_pacf(monthly.dropna(), lags=min(20, len(monthly)-1), ax=ax, method='ywm')
                ax.set_title("PACF")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.warning(f"Could not plot ACF/PACF: {e}")
        
        # ARIMA Model
        st.markdown("---")
        st.markdown("### 🎯 ARIMA Model Fit")
        
        train_size = int(len(monthly) * 0.8)
        train, test = monthly[:train_size], monthly[train_size:]
        
        # Auto-select best order
        with st.spinner("Searching for best ARIMA order by AIC..."):
            best_aic = np.inf
            best_order = (1, suggested_d, 1)
            results = []
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            m = ARIMA(train, order=(p, d, q))
                            fit = m.fit()
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                            results.append({'order': (p, d, q), 'aic': fit.aic})
                        except Exception:
                            continue
        
        st.success(f"✅ Best ARIMA order selected: **ARIMA{best_order}** (AIC = {best_aic:.2f})")
        
        # Fit final model
        try:
            model = ARIMA(train, order=best_order)
            fitted = model.fit()
            
            with st.expander("📋 ARIMA Model Summary"):
                st.text(str(fitted.summary()))
            
            # Forecast
            forecast_test = fitted.forecast(steps=len(test))
            forecast_test.index = test.index
            
            future_dates = pd.date_range(
                start=monthly.index[-1] + pd.DateOffset(months=1),
                periods=arima_forecast_steps, freq='MS'
            )
            full_model = ARIMA(monthly, order=best_order).fit()
            future_forecast = full_model.forecast(steps=arima_forecast_steps)
            future_forecast.index = future_dates
            
            # Metrics
            rmse = float(np.sqrt(mean_squared_error(test.values, forecast_test.values)))
            mae = float(mean_absolute_error(test.values, forecast_test.values))
            mape = float(np.mean(np.abs((test.values - forecast_test.values) / np.maximum(np.abs(test.values), 1))) * 100)
            
            st.markdown("### 📊 Evaluation Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("MAPE", f"{mape:.1f}%")
            
            # Plot
            st.markdown("### 📉 Forecast vs Actual")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly.index, monthly.values, label='Actual', color='#1B5E20', marker='o', markersize=4)
            ax.plot(forecast_test.index, forecast_test.values, label='ARIMA Test Forecast', 
                    color='#2196F3', linestyle='--', marker='x')
            ax.plot(future_forecast.index, future_forecast.values, label=f'Future Forecast ({arima_forecast_steps}m)',
                    color='#FF9800', linestyle='--', marker='s')
            ax.axvline(x=test.index[0], color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
            ax.set_ylabel("Number of Cases")
            ax.set_title(f"ARIMA{best_order} Forecast of Monthly Dengue Cases")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Forecast table
            st.markdown("### 📋 Forecast Values")
            forecast_table = pd.DataFrame({
                'Forecast Month': future_forecast.index.strftime('%Y-%m'),
                'Predicted Cases': np.maximum(future_forecast.values, 0).round(0).astype(int)
            })
            st.dataframe(forecast_table, use_container_width=True)
            
            st.download_button(
                "📥 Download Forecast CSV",
                forecast_table.to_csv(index=False).encode('utf-8'),
                "bantay_dengue_arima_forecast.csv", "text/csv"
            )
            
        except Exception as e:
            st.error(f"ARIMA fitting failed: {e}")


# ------------------------------------------------------------
# TAB 3: GEOGRAPHIC CLUSTERING (K-MEANS)
# ------------------------------------------------------------
with tab3:
    st.header("🗺️ Geographic Clustering (K-Means)")
    st.caption("Chapter 3: Geographic Clustering — Identifying areas with epidemic burden at municipal and barangay levels.")
    
    # Prepare features for clustering
    st.markdown("### Clustering at Barangay Level")
    st.caption("Each barangay is described by its dengue case profile. Clusters reveal similar-risk areas.")
    
    # Create pivot table by barangay
    barangay_features = df_f.groupby(['Municipality', 'Barangay']).agg(
        Total_Cases=('Year', 'count'),
        Mean_Age=('AgeYears', 'mean'),
        Severe_Cases=('Severity', lambda x: (x == 'High').sum()),
        With_Warning=('Clinical_Classification', lambda x: 'Warning' in str(x).sum()),
    ).fillna(0)
    
    # Add temporal features
    for season in ['Dry Season', 'Wet Season']:
        season_counts = df_f[df_f['Season'] == season].groupby(['Municipality', 'Barangay']).size()
        barangay_features[f'Cases_{season.replace(" ", "_")}'] = season_counts
    
    barangay_features = barangay_features.fillna(0)
    
    # Add clinical classification counts
    for clin in df_f['Clinical_Classification'].unique():
        if pd.notna(clin):
            clin_counts = df_f[df_f['Clinical_Classification'] == clin].groupby(['Municipality', 'Barangay']).size()
            barangay_features[f'Clin_{clin[:20]}'] = clin_counts
    
    barangay_features = barangay_features.fillna(0)
    
    if len(barangay_features) < k_clusters * 2:
        st.warning("Not enough barangays for clustering. Adjust filters.")
    else:
        # Scale features
        X_clust = barangay_features.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)
        
        # K-Means
        st.markdown("---")
        st.subheader(f"🔵 K-Means Clustering (K = {k_clusters})")
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)
        sil_km = silhouette_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", k_clusters)
        c2.metric("Silhouette Score", f"{sil_km:.4f}")
        c3.metric("Barangays Clustered", len(barangay_features))
        
        # Interpretation
        if sil_km > 0.5:
            st.success(f"✅ Silhouette Score: {sil_km:.4f} — Good clustering quality.")
        elif sil_km > 0.25:
            st.info(f"ℹ️ Silhouette Score: {sil_km:.4f} — Acceptable clustering quality.")
        else:
            st.warning(f"⚠️ Silhouette Score: {sil_km:.4f} — Weak clustering. Consider different K.")
        
        # Add cluster labels
        barangay_features['Cluster'] = labels_km
        
        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cluster Sizes**")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts_km = pd.Series(labels_km).value_counts().sort_index()
            colors_km = plt.cm.Set2(np.linspace(0, 1, k_clusters))
            ax.bar([f"Cluster {i}" for i in counts_km.index], counts_km.values,
                   color=colors_km, edgecolor='black')
            ax.set_ylabel("Number of Barangays")
            ax.set_title("Barangays per Cluster")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**PCA Visualization**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels_km, cmap='tab10',
                            s=60, alpha=0.8, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Cluster profiles
        st.markdown("**Cluster Profiles (mean feature values):**")
        profile = barangay_features.groupby('Cluster').mean(numeric_only=True).round(2)
        st.dataframe(profile, use_container_width=True)
        
        # Top barangays per cluster
        st.markdown("**🚨 Top Barangays per Cluster (Hotspots):**")
        for c in sorted(barangay_features['Cluster'].unique()):
            sub = barangay_features[barangay_features['Cluster'] == c].sort_values('Total_Cases', ascending=False).head(5)
            st.markdown(f"**Cluster {c} — Top 5 Hotspot Barangays**")
            display_df = sub[['Total_Cases']].copy()
            display_df.index = [f"{m} – {b}" for m, b in display_df.index]
            st.dataframe(display_df, use_container_width=True)
        
        # Municipality-level clustering
        st.markdown("---")
        st.subheader("🏘️ Municipality-Level Clustering")
        
        muni_features = df_f.groupby('Municipality').agg(
            Total_Cases=('Year', 'count'),
            Mean_Age=('AgeYears', 'mean'),
            Severe_Cases=('Severity', lambda x: (x == 'High').sum()),
        ).fillna(0)
        
        if len(muni_features) >= 3:
            scaler_m = StandardScaler()
            X_muni = scaler_m.fit_transform(muni_features)
            
            kmeans_m = KMeans(n_clusters=min(3, len(muni_features)), random_state=42, n_init=10)
            labels_m = kmeans_m.fit_predict(X_muni)
            muni_features['Cluster'] = labels_m
            
            fig, ax = plt.subplots(figsize=(10, 5))
            muni_features_sorted = muni_features.sort_values('Total_Cases', ascending=True)
            colors_m = plt.cm.Reds(np.linspace(0.3, 1, len(muni_features_sorted)))
            ax.barh(muni_features_sorted.index, muni_features_sorted['Total_Cases'], 
                    color=colors_m, edgecolor='black')
            ax.set_xlabel("Total Dengue Cases")
            ax.set_title("Municipality Ranking by Dengue Cases")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ------------------------------------------------------------
# TAB 4: DEMOGRAPHIC PROFILING
# ------------------------------------------------------------
with tab4:
    st.header("👥 Demographic Profiling")
    st.caption("Chapter 3: Demographic Profiling — Characterizing dengue cases by age, sex, and clinical presentation.")
    
    # Age Group Analysis
    st.markdown("### 👶 Dengue Cases by Age Group")
    age_group_counts = df_f['Age_Group'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_age = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#C62828']
        ax.bar(age_group_counts.index, age_group_counts.values, 
               color=colors_age[:len(age_group_counts)], edgecolor='black')
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Number of Cases")
        ax.set_title("Dengue Cases by Age Group (WHO Classification)")
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.dataframe(age_group_counts.rename('Cases').to_frame().assign(
            Percentage=lambda d: (d['Cases']/d['Cases'].sum()*100).round(2)
        ), use_container_width=True)
    
    # Sex Analysis
    st.markdown("---")
    st.markdown("### 👫 Dengue Cases by Sex")
    sex_counts = df_f['Sex'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_sex = ['#42A5F5', '#EC407A']
        ax.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
               colors=colors_sex, startangle=90)
        ax.set_title("Dengue Cases by Sex")
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.dataframe(sex_counts.rename('Cases').to_frame().assign(
            Percentage=lambda d: (d['Cases']/d['Cases'].sum()*100).round(2)
        ), use_container_width=True)
    
    # Age x Sex Analysis
    st.markdown("---")
    st.markdown("### 📊 Age Group × Sex Analysis")
    
    age_sex = df_f.groupby(['Age_Group', 'Sex']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    age_sex.plot(kind='bar', ax=ax, color=['#EC407A', '#42A5F5'], edgecolor='black')
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Dengue Cases by Age Group and Sex")
    ax.legend(title='Sex')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Clinical Classification by Age Group
    st.markdown("---")
    st.markdown("### 🏥 Clinical Severity by Age Group")
    
    sev_age = df_f.groupby(['Age_Group', 'Severity']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_sev = {'High': '#C62828', 'Medium': '#FF9800', 'Low': '#4CAF50'}
    sev_age.plot(kind='bar', ax=ax, color=[colors_sev.get(c, '#888') for c in sev_age.columns], 
                 edgecolor='black')
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Clinical Severity by Age Group")
    ax.legend(title='Severity')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Top Barangays with Most Cases
    st.markdown("---")
    st.markdown("### 🏘️ Top 15 Barangays by Dengue Cases")
    
    top_brgy = df_f.groupby(['Municipality', 'Barangay']).size().sort_values(ascending=False).head(15)
    top_brgy.index = [f"{m} – {b}" for m, b in top_brgy.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_brgy.sort_values().plot(kind='barh', color='#1B5E20', edgecolor='black', ax=ax)
    ax.set_xlabel("Number of Cases")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ------------------------------------------------------------
# TAB 5: CLASSIFICATION (RANDOM FOREST)
# ------------------------------------------------------------
with tab5:
    st.header("🔬 Classification (Random Forest)")
    st.caption("Chapter 3: Random Forest Classification — Predicting dengue severity based on demographic and clinical features.")
    
    # Prepare data for classification
    st.markdown("### 🎯 Predicting Dengue Severity")
    st.caption("Target: Clinical Classification (Dengue Without Warning Signs, Dengue With Warning Signs, Severe Dengue)")
    
    # Select features
    feature_cols = ['AgeYears', 'MORBIDITY WEEK']
    available_features = [c for c in feature_cols if c in df_f.columns]
    
    # Add encoded categorical features
    df_rf = df_f.copy()
    
    # Encode Sex
    le_sex = LabelEncoder()
    df_rf['Sex_Encoded'] = le_sex.fit_transform(df_rf['Sex'].fillna('Unknown'))
    available_features.append('Sex_Encoded')
    
    # Encode Municipality
    le_muni = LabelEncoder()
    df_rf['Municipality_Encoded'] = le_muni.fit_transform(df_rf['Municipality'].fillna('Unknown'))
    available_features.append('Municipality_Encoded')
    
    # Encode Season
    le_season = LabelEncoder()
    df_rf['Season_Encoded'] = le_season.fit_transform(df_rf['Season'].fillna('Unknown'))
    available_features.append('Season_Encoded')
    
    # Target: Clinical Classification
    target_col = 'Clinical_Classification'
    
    # Remove rows with missing target or features
    df_rf = df_rf.dropna(subset=available_features + [target_col])
    
    if len(df_rf) < 50:
        st.warning("Not enough data for classification. Adjust filters.")
    else:
        X = df_rf[available_features]
        y = df_rf[target_col]
        
        # Check class distribution
        st.markdown("**Class Distribution:**")
        class_dist = y.value_counts()
        st.dataframe(class_dist.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        st.markdown("---")
        st.markdown("### 🌲 Random Forest Model")
        
        with st.spinner("Training Random Forest classifier..."):
            rf = RandomForestClassifier(
                n_estimators=rf_estimators,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        st.markdown("### 📊 Evaluation Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Precision", f"{precision:.4f}")
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")
        
        # Interpretation
        if accuracy > 0.8:
            st.success(f"✅ Accuracy: {accuracy:.2%} — Excellent model performance.")
        elif accuracy > 0.6:
            st.info(f"ℹ️ Accuracy: {accuracy:.2%} — Acceptable model performance.")
        else:
            st.warning(f"⚠️ Accuracy: {accuracy:.2%} — Model may need improvement.")
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df['Feature'], importance_df['Importance'], 
                color='#1B5E20', edgecolor='black')
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.dataframe(importance_df.sort_values('Importance', ascending=False), 
                     use_container_width=True)
        
        # Confusion Matrix
        st.markdown("---")
        st.markdown("### 📊 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=rf.classes_, yticklabels=rf.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Public Health Insights")
    st.caption("Chapter 1: Significance of the Study — Providing data-driven insights for PHO-ADS and local health authorities.")
    
    # Get top municipalities and barangays
    top_munis = df_f['Municipality'].value_counts().head(5)
    top_brgys = df_f.groupby(['Municipality', 'Barangay']).size().sort_values(ascending=False).head(10)
    top_age = df_f['Age_Group'].value_counts().idxmax()
    peak_season = df_f['Season'].value_counts().idxmax()
    severe_rate = (df_f['Severity'] == 'High').sum() / len(df_f) * 100
    
    st.markdown(f"""
    ### 📊 Analysis Summary
    
    Based on **{len(df_f):,}** dengue cases from **{df_f['Municipality'].nunique()}** municipalities 
    covering **{year_range[0]}–{year_range[1]}**, using data mining techniques (ARIMA, K-Means, 
    Random Forest) as specified in Chapter 3:
    
    ---
    
    #### 1. For the Provincial Health Office - Agusan del Sur (PHO-ADS)
    
    - **Priority Municipalities:** Focus surveillance and vector control efforts on 
      **{', '.join(top_munis.index[:5])}**, which account for the highest case burden.
    
    - **Hotspot Barangays:** Deploy targeted interventions in these top 10 barangays:
      {chr(10).join([f"  - {m} – {b}" for m, b in top_brgys.index[:10]])}
    
    - **Seasonal Planning:** Dengue cases peak during the **{peak_season}**, 
      with a surge typically observed in **Morbidity Weeks 25-40**. 
      Intensify mosquito control activities 2-4 weeks before this period.
    
    - **High-Risk Demographics:** The most affected age group is **{top_age}**. 
      Target health education campaigns in schools and communities for this group.
    
    ---
    
    #### 2. For Local Government Units (LGUs)
    
    - **Resource Allocation:** Allocate additional resources (insecticides, larvicides, 
      fogging equipment) to municipalities with the highest case counts.
    
    - **Community Mobilization:** Conduct "4S" campaigns (Search and destroy breeding sites, 
      Self-protection measures, Seek early consultation, Support fogging) in hotspot areas.
    
    - **Early Warning System:** Use the ARIMA forecast to anticipate outbreak periods 
      and mobilize response teams proactively.
    
    ---
    
    #### 3. For Community Residents
    
    - **Prevention:** Eliminate stagnant water in containers, tires, and other potential 
      breeding sites around homes, especially during the **{peak_season}**.
    
    - **Protection:** Use mosquito repellents, wear long-sleeved clothing, and use 
      mosquito nets, particularly during daytime when Aedes mosquitoes are most active.
    
    - **Early Consultation:** Seek medical attention immediately if experiencing 
      symptoms such as high fever, severe headache, body aches, or rash.
    
    ---
    
    #### 4. For Future Researchers
    
    - **Integrate Climate Data:** Incorporate rainfall, temperature, and humidity data 
      to improve ARIMA forecast accuracy.
    
    - **Add GIS Analysis:** Use Moran's I and Getis-Ord Gi* for spatial autocorrelation 
      and hotspot detection at the barangay level.
    
    - **Deep Learning:** Compare Random Forest with LSTM and XGBoost for classification.
    
    - **Real-Time Data:** Develop automated data ingestion from health facilities 
      for real-time surveillance updates.
    
    ---
    
    #### 5. Key Statistics
    
    | Metric | Value |
    |--------|-------|
    | Total Cases Analyzed | {len(df_f):,} |
    | Municipalities Covered | {df_f['Municipality'].nunique()} |
    | Barangays Covered | {df_f['Barangay'].nunique()} |
    | Severe Dengue Rate | {severe_rate:.1f}% |
    | Most Affected Age Group | {top_age} |
    | Peak Season | {peak_season} |
    """)
    
    st.markdown("---")
    st.subheader("📥 Download Summary")
    
    try:
        summary_data = {
            'Metric': ['Total Cases', 'Municipalities', 'Barangays', 'Severe Cases', 'Peak Year'],
            'Value': [
                len(df_f),
                df_f['Municipality'].nunique(),
                df_f['Barangay'].nunique(),
                (df_f['Severity'] == 'High').sum(),
                df_f['Year'].value_counts().idxmax()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.download_button(
            "📊 Download Summary CSV",
            summary_df.to_csv(index=False).encode('utf-8'),
            "bantay_dengue_summary.csv", "text/csv"
        )
    except Exception:
        pass


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🎓 Bantay Dengue: A Data-Driven Monitoring Approach for Dengue Incidence • "
    "Ive Jane B. Sabando & Hazel G. Jopia • Bachelor of Science in Information Systems • "
    "Agusan del Sur State University • 2026"
)