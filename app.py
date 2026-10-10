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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bantay Dengue: Data-Driven Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    .main-header {
        font-size: 2.0rem; font-weight: 700; color: #0D1B4C;
        text-align: center; padding: 0.6rem 0 0.2rem 0;
        letter-spacing: 0.3px;
    }
    .sub-header {
        font-size: 0.98rem; color: #5A6472;
        text-align: center; margin-bottom: 1.6rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A237E 0%, #3949AB 100%);
        padding: 1.15rem; border-radius: 10px; color: #FFFFFF;
        text-align: center; box-shadow: 0 3px 10px rgba(26,35,126,0.18);
        border-bottom: 3px solid #FFB300;
    }
    .metric-card h2 { color: #FFD54F; margin: 0; font-size: 1.7rem; font-weight: 700; }
    .metric-card p  { color: #E8EAF6; margin: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.8px; }

    .output-banner {
        background: linear-gradient(90deg, #FF6F00 0%, #FFB300 100%);
        color: #1A1A1A; padding: 0.85rem 1.2rem;
        border-radius: 8px; margin: 0.4rem 0 0.6rem 0;
        font-weight: 700; font-size: 1.0rem;
        box-shadow: 0 3px 10px rgba(255,111,0,0.22);
    }
    .output-banner small {
        display: block; font-weight: 500; opacity: 0.85;
        font-size: 0.82rem; margin-top: 0.2rem; color: #2C2C2C;
    }
    .expected-box {
        background: #F5F7FA; border-left: 5px solid #1A237E;
        padding: 0.7rem 1rem; border-radius: 6px;
        margin-bottom: 1.1rem; font-size: 0.88rem; color: #29304D;
    }
    .expected-box b { color: #1A237E; }

    .alert-card {
        padding: 1rem 1.2rem; border-radius: 8px;
        margin-bottom: 0.75rem; border-left: 6px solid;
        background: #FAFBFC;
    }
    .alert-critical { border-color: #C62828; }
    .alert-warning  { border-color: #F9A825; }
    .alert-watch    { border-color: #1565C0; }
    .alert-ok       { border-color: #2E7D32; }
    .alert-card h5 { margin: 0 0 0.3rem 0; color: #0D1B4C; }
    .alert-card p  { margin: 0; font-size: 0.88rem; color: #444; }

    .finding-card {
        background: #FAFBFC; border: 1px solid #E4E7EE;
        border-left: 6px solid #FF6F00;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .finding-card h5 { margin: 0 0 0.4rem 0; color: #1A237E; font-weight: 700; }
    .finding-card p  { margin: 0; font-size: 0.9rem; color: #333; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEFF4; border-radius: 6px 6px 0 0;
        padding: 10px 18px; font-weight: 600; color: #1A237E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A237E !important; color: #FFD54F !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# OUTPUT BANNERS (Conceptual Framework + Chapter 3)
# ============================================================
def render_output_banner(output_title, chapter3_expected, description):
    st.markdown(
        f'<div class="output-banner">Conceptual Framework Output: {output_title}'
        f'<small>{description}</small></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="expected-box"><b>Chapter 3 Expected Outcome:</b> {chapter3_expected}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=True)
def load_dengue_data():
    search_paths = ['.', '/content/data', '/content', '/mount/src']
    possible_filenames = [
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.xlsx',
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.xls',
        'FINAL-ADS-DENGUE-DATABASE-2021-2025.csv',
        'dengue_data.xlsx', 'dengue_data.csv',
    ]

    filepath = None
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        for fname in possible_filenames:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                filepath = full_path
                break
        if filepath:
            break

    if filepath is None:
        for path in search_paths:
            if not os.path.isdir(path):
                continue
            for ext in ('*.xlsx', '*.xls', '*.csv'):
                for f in glob.glob(os.path.join(path, ext)):
                    base = os.path.basename(f).lower()
                    if 'dengue' in base or 'final-ads' in base or 'bantay' in base:
                        filepath = f
                        break
                if filepath:
                    break
            if filepath:
                break

    if filepath is None:
        return None, "No dengue dataset file found."

    try:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        return df, f"Loaded: {os.path.basename(filepath)} ({len(df):,} rows x {df.shape[1]} cols)"
    except Exception as e:
        return None, f"Error loading file: {e}"


# ============================================================
# COLUMN RESOLVER
# ============================================================
def _resolve_column(df, candidates):
    normalized = {c.lower().replace(' ', '').replace('_', ''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '')
        if key in normalized:
            return normalized[key]
    return None


# ============================================================
# TEXT CLEANING HELPERS
# ============================================================
NULL_TOKENS = {
    '', 'nan', 'none', 'n/a', 'na', 'null', '-', '--', '.', '?',
    'undefined', '#n/a', '#value!', 'not applicable', 'notavailable'
}

def clean_text(value):
    """Trim, collapse whitespace, strip invisible characters, standardize nulls."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = str(value)
    s = s.replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    return s


def clean_titlecase(value):
    s = clean_text(value)
    if pd.isna(s):
        return np.nan
    # Collapse repeated punctuation and re-title
    s = re.sub(r'[^\w\s\.\-\(\)\/,&\']', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.title()


def clean_upper(value):
    s = clean_text(value)
    return s.upper() if isinstance(s, str) else np.nan


def normalize_sex(value):
    s = clean_text(value)
    if pd.isna(s):
        return 'Unknown'
    s = s.lower()
    if s.startswith('m'):
        return 'Male'
    if s.startswith('f'):
        return 'Female'
    return 'Unknown'


def normalize_clinical(value):
    s = clean_text(value)
    if pd.isna(s):
        return 'Unknown'
    s = s.lower()
    if 'severe' in s:
        return 'Severe Dengue'
    if 'warning' in s:
        return 'Dengue With Warning Signs'
    if 'without' in s:
        return 'Dengue Without Warning Signs'
    if 'dengue' in s:
        return 'Dengue (Unspecified)'
    return 'Unknown'


def normalize_outcome(value):
    s = clean_text(value)
    if pd.isna(s):
        return 'Unknown'
    s = s.lower()
    if 'died' in s or 'dead' in s or 'deceased' in s:
        return 'Died'
    if 'alive' in s or 'recovered' in s:
        return 'Alive'
    if 'medical advice' in s or 'against' in s or 'hama' in s:
        return 'Against Medical Advice'
    return 'Unknown'


def normalize_case_class(value):
    s = clean_text(value)
    if pd.isna(s):
        return 'Unknown'
    s = s.lower()
    if 'confirm' in s:
        return 'Confirmed'
    if 'probable' in s:
        return 'Probable'
    if 'suspect' in s:
        return 'Suspect'
    return 'Unknown'


# ============================================================
# PREPROCESSING (Chapter 3 - KDD Process with Cleaning)
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess_data(df):
    df = df.copy()

    # ---------- 1. Normalize column headers (strip whitespace) ----------
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

    col_year     = _resolve_column(df, ['Year'])
    col_mw       = _resolve_column(df, ['MORBIDITY WEEK', 'MorbidityWeek', 'MW'])
    col_caseid   = _resolve_column(df, ['CASE ID', 'CaseID'])
    col_age      = _resolve_column(df, ['AgeYears', 'Age'])
    col_sex      = _resolve_column(df, ['Sex'])
    col_muni     = _resolve_column(df, ['(Permanent Address) City / Municipality',
                                        'Municipality', 'City_Municipality'])
    col_brgy     = _resolve_column(df, ['(Permanent Address) Barangay', 'Barangay'])
    col_clin     = _resolve_column(df, ['ClinClass', 'Clinical Classification',
                                        'Clinical_Classification'])
    col_case     = _resolve_column(df, ['Case Classification', 'Case_Classification'])
    col_outcome  = _resolve_column(df, ['OUTCOME', 'Outcome'])
    col_onset    = _resolve_column(df, ['DOnset', 'Date Onset', 'Date_Onset'])
    col_consult  = _resolve_column(df, ['Date Consulted', 'DateConsulted'])
    col_admit    = _resolve_column(df, ['DAdmit', 'Date Admitted'])
    col_died     = _resolve_column(df, ['DateDied', 'Date Died'])
    col_dob      = _resolve_column(df, ['DOB', 'DateOfBirth'])

    clean = pd.DataFrame()

    # ---------- 2. Numeric columns ----------
    clean['Year'] = pd.to_numeric(
        df[col_year].apply(clean_text), errors='coerce'
    ) if col_year else np.nan

    clean['MORBIDITY WEEK'] = pd.to_numeric(
        df[col_mw].apply(clean_text), errors='coerce'
    ) if col_mw else np.nan

    clean['AgeYears'] = pd.to_numeric(
        df[col_age].apply(clean_text), errors='coerce'
    ) if col_age else np.nan

    # ---------- 3. Text categorical columns ----------
    clean['Case_ID']   = df[col_caseid].apply(clean_text) if col_caseid else np.nan
    clean['Sex']       = df[col_sex].apply(normalize_sex) if col_sex else 'Unknown'
    clean['Municipality'] = df[col_muni].apply(clean_titlecase) if col_muni else np.nan
    clean['Barangay']     = df[col_brgy].apply(clean_titlecase) if col_brgy else np.nan
    clean['Clinical_Classification'] = df[col_clin].apply(normalize_clinical) if col_clin else 'Unknown'
    clean['Case_Classification']     = df[col_case].apply(normalize_case_class) if col_case else 'Unknown'
    clean['Outcome']                 = df[col_outcome].apply(normalize_outcome) if col_outcome else 'Unknown'

    # ---------- 4. Date columns ----------
    for src_col, new_col in [
        (col_onset,   'Date_Onset'),
        (col_consult, 'Date_Consulted'),
        (col_admit,   'Date_Admitted'),
        (col_died,    'Date_Died'),
        (col_dob,     'Date_Of_Birth')
    ]:
        clean[new_col] = pd.to_datetime(df[src_col], errors='coerce') if src_col else pd.NaT

    # ---------- 5. Drop records missing required fields ----------
    clean = clean.dropna(subset=['Year', 'MORBIDITY WEEK', 'AgeYears'])

    # ---------- 6. Valid-range filtering ----------
    clean = clean[(clean['AgeYears'] >= 0) & (clean['AgeYears'] <= 120)]
    clean = clean[(clean['MORBIDITY WEEK'] >= 1) & (clean['MORBIDITY WEEK'] <= 53)]
    clean = clean[(clean['Year'] >= 2000) & (clean['Year'] <= 2100)]

    # ---------- 7. Fill missing categorical labels ----------
    for col in ['Municipality', 'Barangay']:
        clean[col] = clean[col].fillna('Unknown')

    # ---------- 8. Remove duplicate records (by Case_ID when available, else full row) ----------
    before = len(clean)
    if 'Case_ID' in clean.columns and clean['Case_ID'].notna().any():
        clean = clean.drop_duplicates(subset=['Case_ID'], keep='first')
    clean = clean.drop_duplicates()
    removed = before - len(clean)

    # ---------- 9. Derived fields ----------
    bins = [0, 4, 9, 17, 59, 200]
    labels = ['0-4', '5-9', '10-17', '18-59', '60+']
    clean['Age_Group'] = pd.cut(clean['AgeYears'], bins=bins, labels=labels, right=True)

    clean['Month']   = clean['Date_Onset'].dt.month
    clean['Quarter'] = clean['Date_Onset'].dt.quarter

    def get_season(month):
        if pd.isna(month):
            return 'Unknown'
        m = int(month)
        return 'Dry Season' if m in (12, 1, 2, 3, 4, 5) else 'Wet Season'
    clean['Season'] = clean['Month'].apply(get_season)

    def severity_from_clin(c):
        if c == 'Severe Dengue':
            return 'High'
        if c == 'Dengue With Warning Signs':
            return 'Medium'
        if c == 'Dengue Without Warning Signs':
            return 'Low'
        return 'Unknown'
    clean['Severity'] = clean['Clinical_Classification'].apply(severity_from_clin)

    clean.attrs['rows_removed_duplicate'] = removed
    clean.attrs['rows_final'] = len(clean)

    return clean


# ============================================================
# LOAD & PREPROCESS
# ============================================================
df_raw, load_message = load_dengue_data()

st.markdown(
    '<div class="main-header">Bantay Dengue: A Data-Driven Monitoring Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">A Data-Driven Monitoring Approach for Dengue Incidence in '
    'Agusan del Sur<br><i>Provincial Health Office - Agusan del Sur (PHO-ADS) | 2021-2026</i></div>',
    unsafe_allow_html=True
)

if df_raw is None:
    st.error(f"Error: {load_message}")
    st.stop()

st.success(f"{load_message}")

df = preprocess_data(df_raw)

if len(df) < 10:
    st.error("Not enough valid rows after preprocessing.")
    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Dashboard Controls")

muni_options = sorted([m for m in df['Municipality'].dropna().unique() if m != 'Unknown'])
default_munis = muni_options[:5] if len(muni_options) > 5 else muni_options

municipality_filter = st.sidebar.multiselect(
    "Filter by Municipality:", options=muni_options, default=default_munis
)

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
k_clusters = st.sidebar.slider("K-Means Clusters", 2, 8, 4)
arima_forecast_steps = st.sidebar.slider("ARIMA Forecast Horizon (months)", 3, 24, 6)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Cleaning Summary")
st.sidebar.markdown(
    f"""
    - Rows after cleaning: **{len(df):,}**
    - Duplicates removed: **{df.attrs.get('rows_removed_duplicate', 0):,}**
    - Missing rows dropped: **{len(df_raw) - len(df):,}**
    """
)

if not municipality_filter:
    st.warning("Please select at least one municipality.")
    st.stop()

df_f = df[
    (df['Municipality'].isin(municipality_filter)) &
    (df['Year'].between(year_range[0], year_range[1]))
].copy()

if len(df_f) < 10:
    st.warning("Too few records after filtering.")
    st.stop()


# ============================================================
# TABS - aligned with the 4 conceptual-framework outputs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Dengue Hotspot Maps",
    "Bantay Dengue Dashboard",
    "Early Warning Insights",
    "Decision-Support Reports",
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("Executive Overview")
    st.caption("Baseline context for all downstream conceptual-framework outputs.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><p>Total Dengue Cases</p><h2>{len(df_f):,}</h2></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><p>Municipalities</p><h2>{df_f["Municipality"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        f'<div class="metric-card"><p>Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )
    severe_count = int((df_f['Severity'] == 'High').sum())
    c4.markdown(
        f'<div class="metric-card"><p>Severe Cases</p><h2>{severe_count:,}</h2></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("Dengue Incidence by Municipality")
    mun_counts = df_f['Municipality'].value_counts().head(10)
    if len(mun_counts) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        mun_counts.plot(kind='bar', color='#1A237E', edgecolor='black', ax=ax)
        ax.set_ylabel("Number of Cases")
        ax.set_title("Dengue Cases per Municipality")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Clinical Classification Breakdown")
    clin_counts = df_f['Clinical_Classification'].value_counts()
    if len(clin_counts) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            palette = ['#FFB300', '#FB8C00', '#C62828', '#3949AB', '#78909C']
            ax.pie(clin_counts.values, labels=clin_counts.index, autopct='%1.1f%%',
                   colors=palette[:len(clin_counts)], startangle=90)
            ax.set_title("Clinical Classification Distribution")
            st.pyplot(fig)
            plt.close()
        with col2:
            st.dataframe(
                clin_counts.rename('Count').to_frame().assign(
                    Percentage=lambda d: (d['Count'] / d['Count'].sum() * 100).round(2)
                ), use_container_width=True
            )

    st.subheader("Case Outcomes")
    outcome_counts = df_f['Outcome'].value_counts()
    if len(outcome_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_out = ['#2E7D32' if x == 'Alive' else '#C62828' if x == 'Died'
                      else '#F9A825' for x in outcome_counts.index]
        outcome_counts.plot(kind='bar', color=colors_out, edgecolor='black', ax=ax)
        ax.set_ylabel("Number of Cases")
        ax.set_title("Case Outcomes")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ------------------------------------------------------------
# TAB 2: DENGUE HOTSPOT MAPS - Output #1
# ------------------------------------------------------------
with tab2:
    render_output_banner(
        "Dengue Hotspot Maps",
        "Identification of spatial concentrations of dengue cases at municipality and "
        "barangay level to guide targeted vector control and resource allocation.",
        "Ranking and K-Means tiering of municipalities and barangays by dengue burden."
    )

    st.header("Dengue Hotspot Maps")

    st.subheader("Municipality-Level Hotspot Ranking")
    st.caption("Municipalities ranked by case burden. Darker bars indicate higher risk.")

    muni_counts = df_f['Municipality'].value_counts().sort_values(ascending=True)
    if len(muni_counts) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(muni_counts) * 0.35)))
        colors_m = plt.cm.YlOrRd(np.linspace(0.25, 1, len(muni_counts)))
        bars = ax.barh(muni_counts.index, muni_counts.values,
                       color=colors_m, edgecolor='black')
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(muni_counts.max() * 0.01, 0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va='center', fontsize=9)
        ax.set_xlabel("Number of Dengue Cases")
        ax.set_title("Municipality Hotspot Ranking (Darker = Higher Burden)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Barangay-Level Hotspot Ranking (Top 15)")
    st.caption("Highest-burden barangays across the current filter.")

    top_brgy = (df_f.groupby(['Municipality', 'Barangay'])
                    .size()
                    .sort_values(ascending=False)
                    .head(15))
    if len(top_brgy) > 0:
        labels = [f"{m} - {b}" for m, b in top_brgy.index]
        fig, ax = plt.subplots(figsize=(10, max(4, len(top_brgy) * 0.4)))
        colors_b = plt.cm.OrRd(np.linspace(0.3, 1, len(top_brgy)))
        bars = ax.barh(labels[::-1], top_brgy.values[::-1],
                       color=colors_b, edgecolor='black')
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(top_brgy.max() * 0.01, 0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va='center', fontsize=9)
        ax.set_xlabel("Number of Dengue Cases")
        ax.set_title("Barangay Hotspot Ranking (Top 15)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("K-Means Hotspot Tiering (Barangay Level)")
    st.caption("Groups barangays into similar-risk tiers using dengue case profiles.")

    agg_dict = {
        'Total_Cases': ('Year', 'count'),
        'Mean_Age': ('AgeYears', 'mean'),
        'Severe_Cases': ('Severity', lambda x: (pd.Series(x) == 'High').sum()),
        'With_Warning': ('Severity', lambda x: (pd.Series(x) == 'Medium').sum()),
    }
    barangay_features = (
        df_f.groupby(['Municipality', 'Barangay'])
            .agg(**agg_dict)
            .fillna(0)
    )

    for season in ['Dry Season', 'Wet Season']:
        sc = (df_f[df_f['Season'] == season]
              .groupby(['Municipality', 'Barangay'])
              .size()
              .rename(f'Cases_{season.replace(" ", "_")}'))
        barangay_features = barangay_features.join(sc, how='left')

    for clin in df_f['Clinical_Classification'].dropna().unique():
        if clin == 'Unknown':
            continue
        safe = re.sub(r'\W+', '_', str(clin))[:30]
        cc = (df_f[df_f['Clinical_Classification'] == clin]
              .groupby(['Municipality', 'Barangay'])
              .size().rename(f'Clin_{safe}'))
        barangay_features = barangay_features.join(cc, how='left')

    barangay_features = barangay_features.fillna(0)

    if len(barangay_features) < k_clusters * 2:
        st.warning(f"Only {len(barangay_features)} barangays - need at least {k_clusters * 2}.")
    else:
        X_clust = barangay_features.select_dtypes(include=[np.number]).copy()
        X_clust = X_clust.loc[:, X_clust.std() > 0]

        if X_clust.shape[1] < 2:
            st.warning("Not enough feature variation for clustering.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clust)
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            labels_km = kmeans.fit_predict(X_scaled)

            sil_km = 0.0
            if len(set(labels_km)) > 1:
                try:
                    sil_km = silhouette_score(X_scaled, labels_km)
                except Exception:
                    sil_km = 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters", k_clusters)
            c2.metric("Silhouette Score", f"{sil_km:.4f}")
            c3.metric("Barangays Mapped", len(barangay_features))

            barangay_features['Cluster'] = labels_km

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                counts_km = pd.Series(labels_km).value_counts().sort_index()
                colors_km = plt.cm.Set2(np.linspace(0, 1, max(k_clusters, 1)))
                ax.bar([f"Cluster {i}" for i in counts_km.index], counts_km.values,
                       color=colors_km[:len(counts_km)], edgecolor='black')
                ax.set_ylabel("Barangays")
                ax.set_title("Barangays per Hotspot Cluster")
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km,
                                cmap='tab10', s=60, alpha=0.85, edgecolors='black')
                plt.colorbar(sc, label='Cluster')
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("Barangays in PCA Space")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("**Top Hotspot Barangays per Cluster**")
            for c in sorted(barangay_features['Cluster'].unique()):
                sub = (barangay_features[barangay_features['Cluster'] == c]
                       .sort_values('Total_Cases', ascending=False).head(5))
                st.markdown(f"**Cluster {c} - Top 5 Barangays**")
                disp = sub[['Total_Cases']].copy()
                disp.index = [f"{m} - {b}" for m, b in disp.index]
                st.dataframe(disp, use_container_width=True)


# ------------------------------------------------------------
# TAB 3: BANTAY DENGUE DASHBOARD (ARIMA) - Output #2
# ------------------------------------------------------------
with tab3:
    render_output_banner(
        "Bantay Dengue Dashboard",
        "Proactive monitoring of dengue trends with time-series forecasting (ARIMA) "
        "to anticipate future case counts and inform advance planning.",
        "Historical monthly incidence, stationarity diagnostics, and forward ARIMA forecasts."
    )

    st.header("Bantay Dengue: Proactive Trends & Forecast")

    ts_df = df_f.dropna(subset=['Date_Onset']).copy()

    if len(ts_df) < 20:
        st.warning("Not enough records with valid onset dates for forecasting.")
    else:
        ts_df['ym'] = ts_df['Date_Onset'].dt.to_period('M').dt.to_timestamp()
        monthly = ts_df.groupby('ym').size().rename('cases').asfreq('MS').fillna(0)

        if len(monthly) < 12:
            st.warning(f"Only {len(monthly)} months of data. Need at least 12.")
        elif monthly.std() == 0:
            st.warning("Series is constant - cannot fit ARIMA.")
        else:
            st.markdown("### Historical Monthly Trend")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(monthly.index, monthly.values, marker='o', color='#1A237E', markersize=4)
            ax.set_ylabel("Cases")
            ax.set_title("Monthly Dengue Incidence")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.markdown("### Stationarity Test (ADF)")
            try:
                adf_result = adfuller(monthly.dropna(), autolag='AIC')
                adf_stat, adf_p = adf_result[0], adf_result[1]
                c1, c2, c3 = st.columns(3)
                c1.metric("ADF Statistic", f"{adf_stat:.4f}")
                c2.metric("p-value", f"{adf_p:.4f}")
                c3.metric("Conclusion", "Stationary" if adf_p < 0.05 else "Non-stationary")
                suggested_d = 0 if adf_p < 0.05 else 1
                if adf_p < 0.05:
                    st.success("Stationary series (p < 0.05).")
                else:
                    st.info("Non-stationary. Differencing (d=1) recommended.")
            except Exception as e:
                st.warning(f"ADF test failed: {e}")
                suggested_d = 1

            st.markdown("---")
            st.markdown("### ACF / PACF")
            col1, col2 = st.columns(2)
            try:
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_acf(monthly.dropna(), lags=min(20, len(monthly) - 1), ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_pacf(monthly.dropna(), lags=min(20, len(monthly) - 1),
                              ax=ax, method='ywm')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.warning(f"ACF/PACF plotting failed: {e}")

            st.markdown("---")
            st.markdown("### ARIMA Forecast")

            train_size = max(int(len(monthly) * 0.8), 8)
            train, test = monthly[:train_size], monthly[train_size:]

            if len(test) == 0:
                st.warning("Not enough data for train/test split.")
            else:
                with st.spinner("Searching best ARIMA order by AIC..."):
                    best_aic = np.inf
                    best_order = (1, suggested_d, 1)
                    for p in range(0, 3):
                        for d in range(0, 2):
                            for q in range(0, 3):
                                try:
                                    m = ARIMA(train, order=(p, d, q))
                                    fit = m.fit()
                                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                                        best_aic = fit.aic
                                        best_order = (p, d, q)
                                except Exception:
                                    continue

                st.success(f"Best model: ARIMA{best_order} (AIC = {best_aic:.2f})")

                try:
                    model = ARIMA(train, order=best_order).fit()
                    with st.expander("ARIMA Model Summary"):
                        st.text(str(model.summary()))

                    forecast_test = model.forecast(steps=len(test))
                    forecast_test.index = test.index

                    full_model = ARIMA(monthly, order=best_order).fit()
                    future_forecast = full_model.forecast(steps=arima_forecast_steps)
                    future_forecast.index = pd.date_range(
                        start=monthly.index[-1] + pd.DateOffset(months=1),
                        periods=arima_forecast_steps, freq='MS')

                    rmse = float(np.sqrt(np.mean((test.values - forecast_test.values) ** 2)))
                    mae = float(np.mean(np.abs(test.values - forecast_test.values)))
                    mape = float(np.mean(np.abs((test.values - forecast_test.values) /
                                                np.maximum(np.abs(test.values), 1))) * 100)

                    st.markdown("### Forecast Accuracy")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", f"{rmse:.2f}")
                    c2.metric("MAE", f"{mae:.2f}")
                    c3.metric("MAPE", f"{mape:.1f}%")

                    st.markdown("### Forecast Visualization")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(monthly.index, monthly.values, label='Actual',
                            color='#1A237E', marker='o', markersize=4)
                    ax.plot(forecast_test.index, forecast_test.values,
                            label='Test Forecast', color='#FF6F00',
                            linestyle='--', marker='x')
                    ax.plot(future_forecast.index, future_forecast.values,
                            label=f'Future Forecast ({arima_forecast_steps}m)',
                            color='#F9A825', linestyle='--', marker='s')
                    ax.axvline(x=test.index[0], color='gray', linestyle=':',
                               alpha=0.7, label='Train/Test Split')
                    ax.set_ylabel("Cases")
                    ax.set_title(f"ARIMA{best_order} - Monthly Forecast")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown("### Forecast Values")
                    ft = pd.DataFrame({
                        'Forecast Month': future_forecast.index.strftime('%Y-%m'),
                        'Predicted Cases': np.maximum(future_forecast.values, 0).round(0).astype(int)
                    })
                    st.dataframe(ft, use_container_width=True)
                    st.download_button(
                        "Download Forecast CSV",
                        ft.to_csv(index=False).encode('utf-8'),
                        "bantay_dengue_forecast.csv", "text/csv")
                except Exception as e:
                    st.error(f"ARIMA fitting failed: {e}")


# ------------------------------------------------------------
# TAB 4: EARLY WARNING INSIGHTS - Output #3
# ------------------------------------------------------------
with tab4:
    render_output_banner(
        "Early Warning Insights",
        "Automated threshold-based alerts that flag unusual weekly case activity per "
        "municipality for timely PHO-ADS response.",
        "Tiered alerts (Watch / Warning / Critical) derived from historical baselines."
    )

    st.header("Early Warning Insights - Proactive PHO-ADS Alerts")
    st.caption("Based on WHO EWARS principles: compare current weekly counts to historical baselines.")

    weekly = (df_f.dropna(subset=['Date_Onset'])
                  .groupby(['Municipality', pd.Grouper(key='Date_Onset', freq='W')])
                  .size().rename('cases').reset_index())

    if len(weekly) < 5:
        st.warning("Not enough weekly data to compute early warning alerts.")
    else:
        alerts = []
        for muni, grp in weekly.groupby('Municipality'):
            grp = grp.sort_values('Date_Onset').reset_index(drop=True)
            if len(grp) < 5:
                continue
            current = grp.iloc[-1]
            history = grp.iloc[:-1]['cases']
            if len(history) < 4:
                continue
            mean_h = history.mean()
            std_h = history.std() if history.std() > 0 else 1.0
            cur_cases = current['cases']
            if cur_cases > mean_h + 2 * std_h:
                level, css = "CRITICAL", "alert-critical"
            elif cur_cases > mean_h + std_h:
                level, css = "WARNING", "alert-warning"
            elif cur_cases > mean_h:
                level, css = "WATCH", "alert-watch"
            else:
                level, css = "NORMAL", "alert-ok"

            alerts.append({
                'Municipality': muni,
                'Latest Week': pd.to_datetime(current['Date_Onset']).strftime('%Y-%m-%d'),
                'Latest Cases': int(cur_cases),
                'Baseline Mean': round(mean_h, 2),
                'Warning (u+s)': round(mean_h + std_h, 2),
                'Critical (u+2s)': round(mean_h + 2 * std_h, 2),
                'Alert Level': level,
                '_css': css
            })

        if not alerts:
            st.info("Not enough municipal history to compute alerts.")
        else:
            alerts_df = pd.DataFrame(alerts).sort_values(
                'Alert Level',
                key=lambda s: s.map({'CRITICAL': 0, 'WARNING': 1, 'WATCH': 2, 'NORMAL': 3})
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Critical", int((alerts_df['Alert Level'] == 'CRITICAL').sum()))
            c2.metric("Warning",  int((alerts_df['Alert Level'] == 'WARNING').sum()))
            c3.metric("Watch",    int((alerts_df['Alert Level'] == 'WATCH').sum()))
            c4.metric("Normal",   int((alerts_df['Alert Level'] == 'NORMAL').sum()))

            st.markdown("---")
            st.subheader("Active Alerts per Municipality")

            for _, row in alerts_df.iterrows():
                st.markdown(f"""
                <div class="alert-card {row['_css']}">
                    <h5>{row['Alert Level']} - {row['Municipality']}</h5>
                    <p><b>Latest week ({row['Latest Week']}):</b> {row['Latest Cases']} cases |
                    <b>Baseline:</b> {row['Baseline Mean']} |
                    <b>Warning threshold:</b> {row['Warning (u+s)']} |
                    <b>Critical threshold:</b> {row['Critical (u+2s)']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Weekly Trend - Select a Municipality")
            muni_pick = st.selectbox("Municipality:", sorted(weekly['Municipality'].unique()))

            grp = weekly[weekly['Municipality'] == muni_pick].sort_values('Date_Onset')
            if len(grp) > 0:
                mean_h = grp['cases'].iloc[:-1].mean()
                std_h = grp['cases'].iloc[:-1].std() if grp['cases'].iloc[:-1].std() > 0 else 1.0

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(grp['Date_Onset'], grp['cases'], marker='o',
                        color='#1A237E', label='Weekly Cases')
                ax.axhline(mean_h, color='#1565C0', linestyle='--',
                           label=f'Baseline mean ({mean_h:.1f})')
                ax.axhline(mean_h + std_h, color='#F9A825', linestyle='--',
                           label=f'Warning (u+s = {mean_h + std_h:.1f})')
                ax.axhline(mean_h + 2 * std_h, color='#C62828', linestyle='--',
                           label=f'Critical (u+2s = {mean_h + 2 * std_h:.1f})')
                ax.set_ylabel("Cases")
                ax.set_title(f"Weekly Early Warning Chart - {muni_pick}")
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("---")
            st.subheader("Download Alerts Report")
            st.download_button(
                "Download Early Warning Alerts (CSV)",
                alerts_df.drop(columns=['_css']).to_csv(index=False).encode('utf-8'),
                "bantay_dengue_early_warnings.csv", "text/csv")


# ------------------------------------------------------------
# TAB 5: DECISION-SUPPORT REPORTS - Output #4
# (findings only - recommendations are for the manuscript)
# ------------------------------------------------------------
with tab5:
    render_output_banner(
        "Decision-Support Reports",
        "Consolidated evidence snapshots derived from the hotspot, trend, and early-warning "
        "outputs to support evidence-based decisions by PHO-ADS and LGU health officers.",
        "Summary reports of key findings. Recommended actions are documented separately "
        "in the manuscript (not on this dashboard)."
    )

    st.header("Decision-Support Reports - Evidence Summary")

    # ---- Key statistics ----
    total_cases = len(df_f)
    top_munis = df_f['Municipality'].value_counts().head(5)
    top_brgys = df_f.groupby(['Municipality', 'Barangay']).size().sort_values(ascending=False).head(10)
    age_counts = df_f['Age_Group'].value_counts()
    top_age = age_counts.idxmax() if len(age_counts) > 0 else 'N/A'
    season_counts = df_f['Season'].value_counts()
    peak_season = season_counts.idxmax() if len(season_counts) > 0 else 'N/A'
    severe_rate = (df_f['Severity'] == 'High').sum() / max(total_cases, 1) * 100
    top_age_count = int(age_counts.max()) if len(age_counts) > 0 else 0

    # ---- Report 1 ----
    st.markdown(f"""
    <div class="finding-card">
        <h5>Report 1 - Priority Municipality Concentration</h5>
        <p><b>Finding:</b> The top 5 municipalities account for
        <b>{int(top_munis.sum()):,}</b> of <b>{total_cases:,}</b> cases
        ({top_munis.sum()/max(total_cases,1)*100:.1f}% of total burden).</p>
        <p><b>Top municipalities:</b> {', '.join(top_munis.index[:5])}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Report 2 ----
    st.markdown("**Top 10 Barangay Hotspots (from Dengue Hotspot Map output):**")
    if len(top_brgys) > 0:
        for i, (m, b) in enumerate(top_brgys.index, 1):
            st.markdown(f"  {i}. **{m} - {b}**")
    st.markdown(f"""
    <div class="finding-card">
        <h5>Report 2 - Barangay-Level Concentration</h5>
        <p><b>Finding:</b> The top 10 barangays concentrate a disproportionate share
        of the province's dengue burden, indicating localized transmission foci.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Report 3 ----
    st.markdown(f"""
    <div class="finding-card">
        <h5>Report 3 - Vulnerable Population Profile</h5>
        <p><b>Finding:</b> The most affected age group is <b>{top_age}</b>
        with <b>{top_age_count:,}</b> cases. Severe dengue accounts for
        <b>{severe_rate:.1f}%</b> of all reported cases.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Report 4 ----
    st.markdown(f"""
    <div class="finding-card">
        <h5>Report 4 - Seasonal Pattern</h5>
        <p><b>Finding:</b> Peak dengue transmission occurs during the
        <b>{peak_season}</b> in Agusan del Sur.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Report 5 ----
    st.markdown(f"""
    <div class="finding-card">
        <h5>Report 5 - Early Warning System Readiness</h5>
        <p><b>Finding:</b> Weekly case counts monitored against baseline thresholds
        (mean, mean+1sigma, mean+2sigma) consistently flag municipalities requiring
        escalated surveillance.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Download summary ----
    st.markdown("---")
    st.subheader("Download Summary Report (Data Only)")
    report_rows = [
        ("Total Cases Analyzed", f"{total_cases:,}"),
        ("Municipalities Covered", df_f['Municipality'].nunique()),
        ("Barangays Covered", df_f['Barangay'].nunique()),
        ("Top 5 Municipalities", ", ".join(top_munis.index[:5])),
        ("Top 10 Barangays", ", ".join([f"{m} - {b}" for m, b in top_brgys.index[:10]])),
        ("Most Affected Age Group", str(top_age)),
        ("Severe Dengue Rate", f"{severe_rate:.2f}%"),
        ("Peak Season", peak_season),
        ("Year Range", f"{year_range[0]}-{year_range[1]}"),
    ]
    report_df = pd.DataFrame(report_rows, columns=['Metric', 'Value'])
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        "Download Summary Report (CSV)",
        report_df.to_csv(index=False).encode('utf-8'),
        "bantay_dengue_summary_report.csv", "text/csv")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Bantay Dengue: A Data-Driven Monitoring Approach for Dengue Incidence | "
    "Ive Jane B. Sabando & Hazel G. Jopia | Bachelor of Science in Information Systems | "
    "Agusan del Sur State University | 2026"
)
