import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

CONFIG = {
    "file_candidates": ["FINAL-ADS-DENGUE-DATABASE-2021-2025.xlsx", "FINAL-ADS-DENGUE-DATABASE-2021-2025.xls", "FINAL-ADS-DENGUE-DATABASE-2021-2025.csv"],
    "search_paths": [".", "/content/data", "/content", "/mount/src"],
    "train_years": [2022, 2023, 2024],
    "test_years": [2025, 2026],
    "arima_horizon": 24,
    "age_bins": [0, 5, 10, 18, 60, 200],
    "age_labels": ["0-4", "5-9", "10-17", "18-59", "60+"],
    "min_minority_class": 30,
    "k_clusters": 4,
    "rf_estimators": 200,
    "random_state": 42,
    "valid_municipalities": [
        "Bunawan", "Bayugan", "Esperanza", "La Paz", "Loreto", 
        "Prosperidad", "Rosario", "San Francisco", "San Luis", 
        "Santa Josefa", "Sibagat", "Talacogon", "Trento", "Veruela"
    ]
}

NULL_TOKENS = {"", "nan", "none", "n/a", "na", "null", "-", "--", ".", "?", "undefined", "#n/a", "#value!", "not applicable", "notavailable"}

def clean_text(value):
    if value is None: return np.nan
    try:
        if pd.isna(value): return np.nan
    except (TypeError, ValueError): pass
    s = str(value).replace("\xa0", " ").replace("\u200b", "").replace("\t", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in NULL_TOKENS: return np.nan
    return s

def clean_title(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    return re.sub(r"\s+", " ", s).strip().title()

def normalize_sex(value):
    s = clean_text(value)
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if s.startswith("m"): return "Male"
    if s.startswith("f"): return "Female"
    return "Unknown"

def normalize_clinical(value):
    s = clean_text(value)
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "severe" in s: return "Severe Dengue"
    if "warning" in s: return "Dengue With Warning Signs"
    if "without" in s: return "Dengue Without Warning Signs"
    if "dengue" in s: return "Dengue (Unspecified)"
    return "Unknown"

def normalize_case_class(value):
    s = clean_text(value)
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "confirm" in s: return "Confirmed"
    if "probable" in s: return "Probable"
    if "suspect" in s: return "Suspect"
    return "Unknown"

def normalize_outcome(value):
    s = clean_text(value)
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "died" in s or "dead" in s or "deceased" in s: return "Died"
    if "advice" in s or "against" in s: return "Against Medical Advice"
    if "alive" in s or "recovered" in s: return "Alive"
    return "Unknown"

def resolve_column(df, candidates):
    normalized = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in normalized: return normalized[key]
    return None

def locate_file():
    for path in CONFIG["search_paths"]:
        if not os.path.isdir(path): continue
        for fname in CONFIG["file_candidates"]:
            full = os.path.join(path, fname)
            if os.path.exists(full): return full
    for path in CONFIG["search_paths"]:
        if not os.path.isdir(path): continue
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for f in glob.glob(os.path.join(path, ext)):
                base = os.path.basename(f).lower()
                if "dengue" in base or "final-ads" in base or "bantay" in base: return f
    return None

@st.cache_data(show_spinner=True)
def load_raw():
    path = locate_file()
    if path is None: return None, "No dataset file found."
    try:
        if path.lower().endswith(".csv"): df = pd.read_csv(path)
        else: df = pd.read_excel(path)
        return df, os.path.basename(path)
    except Exception as exc: return None, f"Unable to read dataset: {exc}"

@st.cache_data(show_spinner=True)
def preprocess(df_raw):
    df = df_raw.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    if "Year" in df.columns: df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]

    c_year = resolve_column(df, ["Year"])
    c_mw = resolve_column(df, ["MORBIDITY WEEK", "MW", "MorbidityWeek"])
    c_age = resolve_column(df, ["AgeYears", "Age"])
    c_sex = resolve_column(df, ["Sex"])
    c_muni = resolve_column(df, ["(Permanent Address) City / Municipality", "Municipality", "City_Municipality"])
    c_brgy = resolve_column(df, ["(Permanent Address) Barangay", "Barangay"])
    c_clin = resolve_column(df, ["ClinClass", "Clinical Classification", "Clinical_Classification"])
    c_case = resolve_column(df, ["Case Classification", "Case_Classification"])
    c_out = resolve_column(df, ["OUTCOME", "Outcome"])
    c_onset = resolve_column(df, ["DOnset", "Date Onset", "Date_Onset"])
    c_caseid = resolve_column(df, ["CASE ID", "CaseID"])

    clean = pd.DataFrame()
    clean["Year"] = pd.to_numeric(df[c_year].apply(clean_text), errors="coerce") if c_year else np.nan
    clean["Morbidity_Week"] = pd.to_numeric(df[c_mw].apply(clean_text), errors="coerce") if c_mw else np.nan
    clean["Age"] = pd.to_numeric(df[c_age].apply(clean_text), errors="coerce") if c_age else np.nan
    clean["Sex"] = df[c_sex].apply(normalize_sex) if c_sex else "Unknown"
    clean["Municipality"] = df[c_muni].apply(clean_title) if c_muni else np.nan
    clean["Barangay"] = df[c_brgy].apply(clean_title) if c_brgy else np.nan
    clean["Clinical"] = df[c_clin].apply(normalize_clinical) if c_clin else "Unknown"
    clean["Case_Class"] = df[c_case].apply(normalize_case_class) if c_case else "Unknown"
    clean["Outcome"] = df[c_out].apply(normalize_outcome) if c_out else "Unknown"
    clean["Date_Onset"] = pd.to_datetime(df[c_onset], errors="coerce") if c_onset else pd.NaT
    clean["Case_ID"] = df[c_caseid].apply(clean_text) if c_caseid else np.nan

    clean = clean.dropna(subset=["Year", "Morbidity_Week", "Age"])
    clean = clean[(clean["Age"] >= 0) & (clean["Age"] <= 120)]
    clean = clean[(clean["Morbidity_Week"] >= 1) & (clean["Morbidity_Week"] <= 53)]
    clean = clean[(clean["Year"] >= 2000) & (clean["Year"] <= 2100)]

    if clean["Case_ID"].notna().any(): clean = clean.drop_duplicates(subset=["Case_ID"], keep="first")
    clean = clean.drop_duplicates()
    clean["Age"] = clean["Age"].fillna(clean["Age"].median())
    clean["Municipality"] = clean["Municipality"].fillna("Unknown")
    clean["Barangay"] = clean["Barangay"].fillna("Unknown")
    clean["Age_Band"] = pd.cut(clean["Age"], bins=CONFIG["age_bins"], labels=CONFIG["age_labels"], right=False)

    def severity_of(c):
        if c == "Severe Dengue": return "High"
        if c == "Dengue With Warning Signs": return "Medium"
        if c == "Dengue Without Warning Signs": return "Low"
        return "Unknown"
    
    clean["Severity"] = clean["Clinical"].apply(severity_of)
    clean["Month"] = clean["Date_Onset"].dt.month
    clean["Month_Year"] = clean["Date_Onset"].dt.to_period("M").dt.to_timestamp()
    clean["Week_Year"] = clean["Date_Onset"].dt.to_period("W").dt.to_timestamp()
    return clean

def pct(part, total): return 0.0 if total == 0 else 100.0 * part / total

def classification_metrics(y_true, y_pred, labels):
    return (accuracy_score(y_true, y_pred), 
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            confusion_matrix(y_true, y_pred, labels=labels))

def build_rf_features(df):
    feature_df = pd.DataFrame(index=df.index)
    feature_df["Age"] = df["Age"].astype(float)
    feature_df["Morbidity_Week"] = df["Morbidity_Week"].astype(float)
    feature_df["Sex"] = df["Sex"]
    feature_df["Municipality"] = df["Municipality"]
    feature_df["Age_Band"] = df["Age_Band"].astype(str)
    feature_df["Month"] = df["Month"].fillna(0).astype(int)
    return pd.get_dummies(feature_df, columns=["Sex", "Municipality", "Age_Band"], drop_first=False)

def run_rf(df, target_col, target_name, train_years, test_years, seed, n_estimators):
    labels_series = df[target_col]
    valid_mask = labels_series.isin(["Confirmed", "Probable", "Suspect"]) if target_col == "Case_Class" else labels_series.isin(["High", "Medium", "Low"])
    data = df[valid_mask].copy()
    if len(data) < 50: return None

    train = data[data["Year"].isin(train_years)].copy()
    test = data[data["Year"].isin(test_years)].copy()
    if len(train) < 30 or len(test) < 10: return None

    X_train = build_rf_features(train)
    X_test = build_rf_features(test).reindex(columns=X_train.columns, fill_value=0)
    y_train, y_test = train[target_col], test[target_col]
    labels = sorted(data[target_col].unique().tolist())
    min_class_count = int(data[target_col].value_counts().min())

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, prec, rec, f1, cm = classification_metrics(y_test, y_pred, labels)
    baseline_class = y_test.value_counts().idxmax()
    baseline_acc = float((y_test == baseline_class).mean())
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    return {"target": target_name, "labels": labels, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "baseline_class": baseline_class, "baseline_acc": baseline_acc, "feature_importance": fi, "n_train": len(train), "n_test": len(test), "min_class_count": min_class_count}

def detect_arima_order(series, seed):
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    fit = ARIMA(series, order=(p, d, q)).fit()
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception: continue
    return best_order, best_aic

@st.cache_data(show_spinner=True)
def compute_all(df_filtered):
    results = {}
    results["n_total"] = len(df_filtered)
    age_counts = df_filtered["Age_Band"].astype(str).value_counts()
    results["age_counts"], results["age_pct"] = age_counts, (age_counts / age_counts.sum() * 100).round(2)
    sex_counts = df_filtered["Sex"].value_counts()
    results["sex_counts"], results["sex_pct"] = sex_counts, (sex_counts / sex_counts.sum() * 100).round(2)
    clin_counts = df_filtered["Clinical"].value_counts()
    results["clinical_counts"], results["clinical_pct"] = clin_counts, (clin_counts / clin_counts.sum() * 100).round(2)
    results["case_counts"] = df_filtered["Case_Class"].value_counts()
    results["age_sex"] = pd.crosstab(df_filtered["Age_Band"].astype(str), df_filtered["Sex"])
    results["clin_age"] = pd.crosstab(df_filtered["Age_Band"].astype(str), df_filtered["Clinical"])
    results["case_age"] = pd.crosstab(df_filtered["Age_Band"].astype(str), df_filtered["Case_Class"])
    results["yearly"] = df_filtered.groupby("Year").size().sort_index()
    results["monthly"] = (df_filtered.dropna(subset=["Date_Onset"]).groupby("Month_Year").size().asfreq("MS").fillna(0))
    results["weekly"] = (df_filtered.dropna(subset=["Date_Onset"]).groupby("Week_Year").size().sort_index())
    results["muni_counts"] = df_filtered["Municipality"].value_counts()
    results["brgy_counts"] = df_filtered.groupby(["Municipality", "Barangay"]).size().sort_values(ascending=False)

    muni_agg = df_filtered.groupby("Municipality").agg(Cases=("Year", "count"), Mean_Age=("Age", "mean"), Severe=("Severity", lambda s: int((s == "High").sum())), Confirmed=("Case_Class", lambda s: int((s == "Confirmed").sum()))).fillna(0)
    cluster_summary = None
    if len(muni_agg) >= CONFIG["k_clusters"]:
        feat = muni_agg.copy()
        scaled = StandardScaler().fit_transform(feat)
        km = KMeans(n_clusters=CONFIG["k_clusters"], random_state=CONFIG["random_state"], n_init=10)
        feat["Cluster"] = km.fit_predict(scaled)
        cluster_summary = feat.sort_values("Cases", ascending=False)
    results["muni_agg"], results["cluster_summary"] = muni_agg, cluster_summary
    results["rf_case"] = run_rf(df_filtered, "Case_Class", "Case Classification", CONFIG["train_years"], CONFIG["test_years"], CONFIG["random_state"], CONFIG["rf_estimators"])
    results["rf_sev"] = run_rf(df_filtered, "Severity", "Severity", CONFIG["train_years"], CONFIG["test_years"], CONFIG["random_state"], CONFIG["rf_estimators"])

    forecast_block = None
    monthly_all = results["monthly"]
    if len(monthly_all) >= 12 and monthly_all.std() > 0:
        train_series = monthly_all[monthly_all.index.year.isin(CONFIG["train_years"])]
        test_series = monthly_all[monthly_all.index.year.isin(CONFIG["test_years"])]
        if len(train_series) >= 8:
            try: adf_p = adfuller(train_series.dropna(), autolag="AIC")[1]
            except Exception: adf_p = 1.0
            order, aic = detect_arima_order(train_series, CONFIG["random_state"])
            try:
                model_train = ARIMA(train_series, order=order).fit()
                test_forecast = model_train.forecast(steps=len(test_series)) if len(test_series) > 0 else None
                rmse = float(np.sqrt(mean_squared_error(test_series.values, test_forecast.values))) if test_forecast is not None and len(test_forecast) == len(test_series) else float("nan")
                full_model = ARIMA(monthly_all, order=order).fit()
                future = full_model.forecast(steps=CONFIG["arima_horizon"])
                future.index = pd.date_range(start=monthly_all.index[-1] + pd.DateOffset(months=1), periods=CONFIG["arima_horizon"], freq="MS")
                forecast_block = {"order": order, "aic": aic, "adf_p": adf_p, "rmse": rmse, "actual": monthly_all, "test_series": test_series, "test_forecast": test_forecast, "future": future}
            except Exception: forecast_block = None
    results["forecast"] = forecast_block

    weekly_muni = (df_filtered.dropna(subset=["Date_Onset"]).groupby(["Municipality", pd.Grouper(key="Date_Onset", freq="W")]).size().rename("cases").reset_index())
    alerts = []
    if len(weekly_muni) > 0:
        for muni, grp in weekly_muni.groupby("Municipality"):
            grp = grp.sort_values("Date_Onset").reset_index(drop=True)
            if len(grp) < 6: continue
            latest = grp.iloc[-1]
            history = grp.iloc[:-1]["cases"]
            mean_h, std_h = history.mean(), history.std() if history.std() > 0 else 1.0
            current = float(latest["cases"])
            if current > mean_h + 2 * std_h: level = "Critical"
            elif current > mean_h + std_h: level = "Warning"
            elif current > mean_h: level = "Watch"
            else: level = "Normal"
            alerts.append({"Municipality": muni, "Latest_Week": pd.to_datetime(latest["Date_Onset"]).strftime("%Y-%m-%d"), "Latest_Cases": int(current), "Baseline_Mean": round(mean_h, 2), "Warning_Threshold": round(mean_h + std_h, 2), "Critical_Threshold": round(mean_h + 2 * std_h, 2), "Alert_Level": level})
    results["alerts"] = pd.DataFrame(alerts)
    return results

st.set_page_config(page_title="Bantay Dengue Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
.title-main { font-size: 1.9rem; font-weight: 700; color: #0D1B4C; text-align: center; padding: 0.4rem 0 0.2rem 0; }
.title-sub { font-size: 0.95rem; color: #5A6472; text-align: center; margin-bottom: 1.4rem; }
.kpi { background: linear-gradient(135deg, #1A237E 0%, #3949AB 100%); padding: 1rem; border-radius: 10px; color: #FFFFFF; text-align: center; border-bottom: 3px solid #FFB300; }
.kpi h2 { color: #FFD54F; margin: 0; font-size: 1.6rem; font-weight: 700; }
.kpi p { color: #E8EAF6; margin: 0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; }
.finding { background: #FAFBFC; border: 1px solid #E4E7EE; border-left: 6px solid #FF6F00; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.9rem; }
.finding h5 { margin: 0 0 0.3rem 0; color: #1A237E; font-weight: 700; }
.finding p { margin: 0; font-size: 0.9rem; color: #333; }
.alert { padding: 0.85rem 1rem; border-radius: 8px; margin-bottom: 0.6rem; border-left: 6px solid; }
.alert-critical { border-color: #C62828; background: #FFF3F3; }
.alert-warning  { border-color: #F9A825; background: #FFFCF2; }
.alert-watch    { border-color: #1565C0; background: #F2F7FF; }
.alert-normal   { border-color: #2E7D32; background: #F2FBF3; }
.alert h5 { margin: 0 0 0.3rem 0; color: #0D1B4C; }
.alert p { margin: 0; font-size: 0.85rem; color: #444; }
.stTabs [data-baseweb="tab-list"] { gap: 12px !important; padding-bottom: 15px !important; }
.stTabs [data-baseweb="tab"] { background-color: #FFFFFF !important; border: 2px solid #1A237E !important; border-radius: 8px !important; padding: 8px 20px !important; color: #1A237E !important; font-weight: 600 !important; box-shadow: 0px 2px 5px rgba(0,0,0,0.1) !important; transition: all 0.2s ease-in-out !important; }
.stTabs [data-baseweb="tab"]:hover { background-color: #F2F7FF !important; }
.stTabs [aria-selected="true"] { background-color: #1A237E !important; color: #FFFFFF !important; border: 2px solid #1A237E !important; box-shadow: 0px 4px 8px rgba(26, 35, 126, 0.4) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv", "xlsx"], help="Limit 200MB per file • CSV, XLSX")
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)
st.sidebar.markdown("---")

st.markdown('<div class="title-main">BANTAY DENGUE DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="title-sub">Upload a CSV/Excel file — explore data cleaning, EDA, clustering, prediction, and forecasting.</div>', unsafe_allow_html=True)

if uploaded_file is None and not use_example:
    st.info("Please upload a CSV/Excel to begin, or toggle 'Use example dataset' in the sidebar.")
    st.stop()

raw = None
source_name = ""
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        source_name = uploaded_file.name
        st.sidebar.success(f"Loaded: {source_name}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()
elif use_example:
    raw, source_name = load_raw()
    if raw is not None: st.sidebar.success(f"Loaded: {source_name}")

if raw is None:
    st.error("Dataset could not be loaded. Please upload a file or check your example dataset path.")
    st.stop()

df = preprocess(raw)

# Strict Geographic Filter: Remove invalid municipalities like "City Of Bayugan"
valid_lower = [m.lower() for m in CONFIG["valid_municipalities"]]
df = df[df["Municipality"].str.lower().isin(valid_lower)]

if len(df) < 50:
    st.error("The dataset has too few usable records after cleaning (or contains no valid Agusan del Sur municipalities).")
    st.stop()

st.sidebar.header("Filters")
muni_options = sorted([m for m in df["Municipality"].dropna().unique() if m != "Unknown"])
picked_munis = st.sidebar.multiselect("Municipality", options=muni_options, default=muni_options[:6] if len(muni_options) > 6 else muni_options)
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
picked_years = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
picked_sex = st.sidebar.multiselect("Sex", options=["Male", "Female", "Unknown"], default=["Male", "Female"])

if not picked_munis or not picked_sex:
    st.warning("Select at least one municipality and one sex category.")
    st.stop()

df_f = df[df["Municipality"].isin(picked_munis) & df["Year"].between(picked_years[0], picked_years[1]) & df["Sex"].isin(picked_sex)].copy()
if len(df_f) < 30:
    st.warning("Too few records after applying filters.")
    st.stop()

res = compute_all(df_f)

tabs = st.tabs(["Overview", "Results by Age Group", "Results by Sex", "Case Classification", "Results Across Time", "Geographic Hotspots", "Model Performance", "Significant Factors", "Forecast", "Decision Support"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi"><p>Total cases</p><h2>{res["n_total"]:,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><p>Municipalities</p><h2>{df_f["Municipality"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><p>Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi"><p>Severe cases</p><h2>{int((df_f["Severity"] == "High").sum()):,}</h2></div>', unsafe_allow_html=True)
    st.markdown("##### Clinical classification distribution")
    clin = res["clinical_counts"]
    if len(clin) > 0:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(clin.values, labels=clin.index, autopct="%1.1f%%", colors=["#FFB300", "#FB8C00", "#C62828", "#3949AB", "#78909C"][:len(clin)], startangle=90)
            ax.set_title("Clinical classification")
            st.pyplot(fig)
            plt.close()
        with col_b:
            table = clin.rename("Count").to_frame()
            table["Percent"] = res["clinical_pct"].values
            st.dataframe(table, use_container_width=True)
    st.markdown("##### Case classification distribution")
    case = res["case_counts"]
    if len(case) > 0:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(case.index, case.values, color="#1A237E", edgecolor="black")
        ax.set_ylabel("Cases")
        ax.set_title("Case classification")
        st.pyplot(fig)
        plt.close()
    st.markdown("##### Outcome distribution")
    outcome_counts = df_f["Outcome"].value_counts()
    outcome_counts = outcome_counts[outcome_counts.index != "Unknown"]
    if len(outcome_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#2E7D32" if x == "Alive" else "#C62828" if x == "Died" else "#F9A825" for x in outcome_counts.index]
        ax.bar(outcome_counts.index, outcome_counts.values, color=colors, edgecolor="black")
        ax.set_ylabel("Cases")
        ax.set_title("Outcome")
        st.pyplot(fig)
        plt.close()

with tabs[1]:
    age = res["age_counts"]
    if len(age) == 0: st.info("No age data available.")
    else:
        st.markdown("##### Cases by age group")
        table = age.rename("Cases").to_frame()
        table["Share %"] = res["age_pct"].values
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(age.index, age.values, color="#1A237E", edgecolor="black")
        ax.set_ylabel("Cases")
        ax.set_title("Age group distribution")
        st.pyplot(fig)
        plt.close()
        crosstab = res["age_sex"]
        if crosstab.shape[1] > 0:
            st.markdown("##### Age group by sex")
            st.dataframe(crosstab, use_container_width=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#3949AB", "#EC407A", "#90A4AE"][:crosstab.shape[1]], edgecolor="black")
            ax.set_ylabel("Cases")
            ax.set_title("Age group by sex")
            ax.legend(title="Sex")
            st.pyplot(fig)
            plt.close()
        clin_ct = res["clin_age"]
        if clin_ct.shape[1] > 0:
            st.markdown("##### Age group by clinical classification")
            st.dataframe(clin_ct, use_container_width=True)
        st.markdown(f'The largest age group is **{age.idxmax()}** with **{int(age.max()):,}** cases ({pct(age.max(), int(age.sum())):.1f}% of the filtered total).')

with tabs[2]:
    sex = res["sex_counts"]
    if len(sex) == 0: st.info("No sex data available.")
    else:
        st.markdown("##### Cases by sex")
        table = sex.rename("Cases").to_frame()
        table["Share %"] = res["sex_pct"].values
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(sex.index, sex.values, color=["#3949AB", "#EC407A", "#90A4AE"][:len(sex)], edgecolor="black")
        ax.set_ylabel("Cases")
        ax.set_title("Sex distribution")
        st.pyplot(fig)
        plt.close()
        crosstab = res["age_sex"]
        if crosstab.shape[1] > 0:
            st.markdown("##### Sex by age group")
            st.dataframe(crosstab.T, use_container_width=True)
        st.markdown(f'**{sex.idxmax()}** accounts for **{int(sex.max()):,}** of {int(sex.sum()):,} filtered cases ({pct(sex.max(), int(sex.sum())):.1f}%).')

with tabs[3]:
    case = res["case_counts"]
    if len(case) == 0: st.info("No case classification data available.")
    else:
        st.markdown("##### Case classification counts")
        table = case.rename("Cases").to_frame()
        table["Share %"] = (case / case.sum() * 100).round(2).values
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(case.index, case.values, color="#1A237E", edgecolor="black")
        ax.set_ylabel("Cases")
        ax.set_title("Case classification")
        st.pyplot(fig)
        plt.close()
        crosstab = res["case_age"]
        if crosstab.shape[1] > 0:
            st.markdown("##### Case classification by age group")
            st.dataframe(crosstab, use_container_width=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#3949AB", "#FFA726", "#C62828"][:crosstab.shape[1]], edgecolor="black")
            ax.set_ylabel("Cases")
            ax.set_title("Case classification by age group")
            st.pyplot(fig)
            plt.close()
        st.markdown(f'Severe dengue represents **{pct(int((df_f["Severity"] == "High").sum()), int(df_f["Severity"].isin(["High", "Medium", "Low"]).sum())):.1f}%** of cases with an assigned severity.')

with tabs[4]:
    yearly = res["yearly"]
    if len(yearly) == 0: st.info("No yearly data available.")
    else:
        st.markdown("##### Cases by year")
        st.dataframe(yearly.rename("Cases"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(yearly.index, yearly.values, marker="o", color="#1A237E")
        ax.set_ylabel("Cases")
        ax.set_title("Yearly incidence")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    monthly = res["monthly"]
    if len(monthly) > 0:
        st.markdown("##### Monthly incidence")
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(monthly.index, monthly.values, color="#3949AB", marker="o", markersize=3)
        ax.set_ylabel("Cases")
        ax.set_title("Monthly incidence")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    weekly = res["weekly"]
    if len(weekly) > 0:
        st.markdown("##### Weekly incidence")
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(weekly.index, weekly.values, color="#FF6F00", linewidth=1)
        ax.set_ylabel("Cases")
        ax.set_title("Weekly incidence")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    if len(monthly) > 0:
        st.markdown(f'Highest monthly count in the filtered range is **{int(monthly.max()):,}** cases, recorded in **{monthly.idxmax().strftime("%B %Y")}**.')

with tabs[5]:
    muni_counts = res["muni_counts"]
    if len(muni_counts) == 0: st.info("No municipality data available.")
    else:
        st.markdown("##### Municipality case counts")
        st.dataframe(muni_counts.rename("Cases"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(muni_counts) * 0.3)))
        ordered = muni_counts.sort_values()
        ax.barh(ordered.index, ordered.values, color=plt.cm.YlOrRd(np.linspace(0.3, 1, len(ordered))), edgecolor="black")
        ax.set_xlabel("Cases")
        ax.set_title("Municipality ranking")
        st.pyplot(fig)
        plt.close()
    brgy_counts = res["brgy_counts"]
    if len(brgy_counts) > 0:
        st.markdown("##### Top 15 barangays")
        top_brgy = brgy_counts.head(15)
        labels = [f"{m} - {b}" for m, b in top_brgy.index]
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(top_brgy) * 0.36)))
        ax.barh(labels[::-1], top_brgy.values[::-1], color=plt.cm.OrRd(np.linspace(0.3, 1, len(top_brgy))), edgecolor="black")
        ax.set_xlabel("Cases")
        ax.set_title("Barangay ranking")
        st.pyplot(fig)
        plt.close()
    cluster_summary = res["cluster_summary"]
    if cluster_summary is not None:
        st.markdown("##### Municipality clusters")
        st.dataframe(cluster_summary.round(2), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        cluster_counts = cluster_summary["Cluster"].value_counts().sort_index()
        ax.bar([f"Cluster {c}" for c in cluster_counts.index], cluster_counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(cluster_counts))), edgecolor="black")
        ax.set_ylabel("Municipalities")
        ax.set_title("Municipalities per cluster")
        st.pyplot(fig)
        plt.close()
        total = int(cluster_summary["Cases"].sum())
        top_muni = cluster_summary.index[0]
        st.markdown(f'The highest-burden municipality is **{top_muni}** with **{int(cluster_summary.loc[top_muni, "Cases"]):,}** cases ({pct(cluster_summary.loc[top_muni, "Cases"], total):.1f}% of the filtered total).')

with tabs[6]:
    if res["rf_case"] is None and res["rf_sev"] is None: st.info("Random Forest could not be trained on the filtered dataset.")
    else:
        if res["rf_case"] is not None:
            r = res["rf_case"]
            st.markdown(f"##### Model: {r['target']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{r['acc']:.3f}")
            c2.metric("Precision (macro)", f"{r['prec']:.3f}")
            c3.metric("Recall (macro)", f"{r['rec']:.3f}")
            c4.metric("F1 (macro)", f"{r['f1']:.3f}")
            st.caption(f"Trained on {r['n_train']:,} records from 2022 to 2024; tested on {r['n_test']:,} records from 2025 to 2026. Majority-class baseline accuracy: {r['baseline_acc']:.3f} (always predicting '{r['baseline_class']}'). Smallest class size: {r['min_class_count']}.")
            st.dataframe(pd.DataFrame(r["cm"], index=[f"actual {x}" for x in r["labels"]], columns=[f"pred {x}" for x in r["labels"]]), use_container_width=True)
        if res["rf_sev"] is not None:
            r = res["rf_sev"]
            st.markdown(f"##### Model: {r['target']}")
            if r["min_class_count"] < CONFIG["min_minority_class"]: st.warning(f"The smallest severity class has only {r['min_class_count']} records. Metrics below should be read as indicative only.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{r['acc']:.3f}")
            c2.metric("Precision (macro)", f"{r['prec']:.3f}")
            c3.metric("Recall (macro)", f"{r['rec']:.3f}")
            c4.metric("F1 (macro)", f"{r['f1']:.3f}")
            st.caption(f"Trained on {r['n_train']:,} records from 2022 to 2024; tested on {r['n_test']:,} records from 2025 to 2026. Majority-class baseline accuracy: {r['baseline_acc']:.3f} (always predicting '{r['baseline_class']}').")
            st.dataframe(pd.DataFrame(r["cm"], index=[f"actual {x}" for x in r["labels"]], columns=[f"pred {x}" for x in r["labels"]]), use_container_width=True)

with tabs[7]:
    any_model = res["rf_case"] is not None or res["rf_sev"] is not None
    if not any_model: st.info("Feature importance requires a trained model.")
    else:
        model_choice = st.selectbox("Model", options=[x for x in ["Case Classification", "Severity"] if res["rf_case" if x == "Case Classification" else "rf_sev"] is not None])
        chosen = res["rf_case"] if model_choice == "Case Classification" else res["rf_sev"]
        fi = chosen["feature_importance"]
        top_fi = fi.head(15)
        st.markdown(f"##### Top 15 features driving {model_choice}")
        st.dataframe(top_fi.rename("Importance").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, max(3.6, len(top_fi) * 0.32)))
        ax.barh(top_fi.index[::-1], top_fi.values[::-1], color="#1A237E", edgecolor="black")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance for {model_choice}")
        st.pyplot(fig)
        plt.close()
        st.markdown(f'The most influential feature for **{model_choice}** is **{top_fi.index[0]}** (importance = {top_fi.iloc[0]:.3f}).')

with tabs[8]:
    fb = res["forecast"]
    if fb is None: st.info("Not enough monthly data to fit an ARIMA model on this filter.")
    else:
        st.markdown("##### Monthly series and forecast")
        c1, c2, c3 = st.columns(3)
        c1.metric("ARIMA order", str(fb["order"]))
        c2.metric("AIC", f"{fb['aic']:.2f}")
        c3.metric("ADF p-value", f"{fb['adf_p']:.4f}")
        st.metric("Test RMSE (2025 to 2026)", f"{fb['rmse']:.2f}" if not np.isnan(fb["rmse"]) else "not available")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(fb["actual"].index, fb["actual"].values, color="#1A237E", marker="o", markersize=3, label="Observed")
        if fb["test_forecast"] is not None and len(fb["test_series"]) > 0:
            ax.plot(fb["test_series"].index, fb["test_forecast"].values, color="#FF6F00", linestyle="--", marker="x", label="Test forecast")
        ax.plot(fb["future"].index, np.maximum(fb["future"].values, 0), color="#F9A825", linestyle="--", marker="s", label=f"Next {CONFIG['arima_horizon']} months")
        ax.set_ylabel("Cases")
        ax.set_title("ARIMA monthly forecast")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
        future_df = pd.DataFrame({"Month": fb["future"].index.strftime("%Y-%m"), "Forecast": np.maximum(fb["future"].values, 0).round(0).astype(int)})
        st.dataframe(future_df, use_container_width=True)
        st.download_button("Download forecast CSV", future_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")

with tabs[9]:
    alerts_df = res["alerts"]
    if alerts_df is None or len(alerts_df) == 0: st.info("Not enough weekly history to compute early warning alerts.")
    else:
        st.markdown("##### Threshold-based early warning status by municipality")
        level_colors = {"Critical": "alert-critical", "Warning": "alert-warning", "Watch": "alert-watch", "Normal": "alert-normal"}
        order = {"Critical": 0, "Warning": 1, "Watch": 2, "Normal": 3}
        alerts_sorted = alerts_df.sort_values("Alert_Level", key=lambda s: s.map(order))
        counts = alerts_sorted["Alert_Level"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical", int(counts.get("Critical", 0)))
        c2.metric("Warning", int(counts.get("Warning", 0)))
        c3.metric("Watch", int(counts.get("Watch", 0)))
        c4.metric("Normal", int(counts.get("Normal", 0)))
        for _, row in alerts_sorted.iterrows():
            st.markdown(f'<div class="alert {level_colors[row["Alert_Level"]]}"><h5>{row["Alert_Level"]} - {row["Municipality"]}</h5><p>Latest week {row["Latest_Week"]}: {row["Latest_Cases"]} cases. Baseline mean {row["Baseline_Mean"]}, warning {row["Warning_Threshold"]}, critical {row["Critical_Threshold"]}.</p></div>', unsafe_allow_html=True)
        st.markdown("##### Municipalities above warning threshold")
        st.dataframe(alerts_sorted[alerts_sorted["Alert_Level"].isin(["Critical", "Warning"])].reset_index(drop=True), use_container_width=True)
        st.download_button("Download alerts CSV", alerts_sorted.to_csv(index=False).encode("utf-8"), "alerts.csv", "text/csv")

st.markdown("---")
st.caption("Bantay Dengue Dashboard - computed results from the Agusan del Sur dengue case dataset.")
