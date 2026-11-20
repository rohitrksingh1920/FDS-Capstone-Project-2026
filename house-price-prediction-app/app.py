from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.model_loader import load_model, MODEL_SEARCH_PATHS
from utils.prediction import FEATURE_COLUMNS, predict_house_price

st.set_page_config(
    page_title="HomeValue AI | Property Intelligence",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
GOLD = "#2FBF85"  # Emerald accent
CHARCOAL = "#101714"  # Obsidian background
SURFACE = "#18221D"
TEXT = "#F2F6F3"
MUTED = "#A9B8AF"

METRICS = {
    "MAE": 8384.713274475082,
    "MSE": 107941927.52511284,
    "RMSE": 10389.510456470643,
    "R2": 0.9983377048996545,
}
CV_RESULTS = pd.DataFrame({
    "Model": ["Linear Regression", "Lasso", "Ridge", "Random Forest"],
    "Mean CV R²": [0.998500, 0.998500, 0.998498, 0.992359],
    "Std CV R²": [0.000203, 0.000203, 0.000200, 0.001286],
})

# IMPORTANT: Keep this CSS inside this st.markdown call.
st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700;800&display=swap');

:root {
    --bg: #101714;
    --surface: #18221D;
    --surface-raised: #1D2A23;
    --emerald: #2FBF85;
    --emerald-bright: #48D69B;
    --emerald-line: rgba(47,191,133,.34);
    --emerald-wash: rgba(47,191,133,.09);
    --text: #F2F6F3;
    --muted: #A9B8AF;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: var(--bg) !important;
}
[data-testid="stSidebar"] {
    background: #131C17 !important;
    border-right: 1px solid var(--emerald-line);
}
[data-testid="stSidebar"] > div {
    padding-top: 2rem;
    height: 100vh;
    overflow-y: auto;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted) !important; }
.main .block-container {
    max-width: 1440px;
    padding: 2.8rem 3rem 3.2rem;
}

/* Typography */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -.45px;
}
p, label, span, small, li { color: var(--muted); }
.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1.1px;
    color: var(--text);
}
.brand-caption {
    font-size: 9px;
    letter-spacing: 2.8px;
    text-transform: uppercase;
    margin-top: 5px;
    color: var(--emerald) !important;
    font-weight: 700;
}
.eyebrow {
    color: var(--emerald) !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.7px;
    text-transform: uppercase;
}
.hero {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(36px, 4.2vw, 58px);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -1.8px;
    color: var(--text);
    margin: 16px 0 14px;
}
.desc {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
    max-width: 820px;
}
.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 26px;
    line-height: 1.3;
    font-weight: 800;
    color: var(--text);
}
.gold-rule {
    height: 3px;
    width: 54px;
    background: var(--emerald);
    border-radius: 4px;
    margin: 12px 0 15px;
}

/* Cards */
.metric-card, .content-card {
    position: relative;
    background: linear-gradient(145deg, rgba(47,191,133,.085), var(--surface) 72%);
    border: 1px solid var(--emerald-line);
    border-radius: 16px;
    padding: 23px;
    height: 100%;
    box-sizing: border-box;
    box-shadow: 0 12px 30px rgba(0,0,0,.12);
}
.metric-card {
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    transition: transform .2s ease, border-color .2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--emerald);
}
.metric-card:before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--emerald), var(--emerald-bright));
}
.metric-label, .micro-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    color: var(--emerald);
    text-transform: uppercase;
}
.metric-value {
    font-size: clamp(20px, 2vw, 28px);
    line-height: 1.28;
    font-weight: 800;
    color: var(--text);
    margin-top: 12px;
    overflow-wrap: anywhere;
}
.metric-note {
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
}
.content-card { padding: 26px; }
.content-card p { line-height: 1.8; color: var(--muted); }
.content-card h3 { color: var(--text) !important; }
.result-card {
    background: linear-gradient(125deg, var(--emerald), #218B63);
    color: #07130D;
    border: 1px solid var(--emerald-bright);
    border-radius: 18px;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 18px 38px rgba(20,160,105,.16);
}
.result-card * { color: #07130D !important; }
.result-price {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(34px, 4vw, 48px);
    font-weight: 800;
    overflow-wrap: anywhere;
    margin-top: 10px;
}

/* Controls */
div.stButton > button, div.stFormSubmitButton > button {
    background: var(--emerald) !important;
    color: #07130D !important;
    border: 1px solid var(--emerald) !important;
    border-radius: 9px !important;
    font-weight: 800 !important;
    min-height: 44px;
    transition: .2s ease;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: var(--emerald-bright) !important;
    color: #07130D !important;
    border-color: var(--emerald-bright) !important;
    transform: translateY(-1px);
}
input, textarea, [data-baseweb="input"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}
[data-baseweb="input"] > div, [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--emerald-line) !important;
    border-radius: 9px !important;
}
[data-testid="stNumberInput"] button {
    background: var(--surface-raised) !important;
    color: var(--emerald) !important;
    border-color: var(--emerald-line) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--emerald) !important;
    color: #07130D !important;
}
[data-testid="stRadio"] label { color: var(--text) !important; }
[data-testid="stRadio"] label:hover { color: var(--emerald) !important; }
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--emerald-line) !important;
    border-radius: 11px !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }

/* Streamlit data grid */
[data-testid="stDataFrame"] {
    --gdg-bg-cell: #18221D;
    --gdg-bg-cell-medium: #1D2A23;
    --gdg-bg-header: #1D2A23;
    --gdg-text-dark: #F2F6F3;
    --gdg-text-medium: #A9B8AF;
    --gdg-border-color: rgba(47,191,133,.28);
    border: 1px solid var(--emerald-line) !important;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stTable"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--emerald-line);
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--emerald-line);
    border-radius: 10px;
}
hr { border-color: var(--emerald-line) !important; }
footer { visibility: hidden; }

@media (max-width: 900px) {
    .main .block-container { padding: 1.3rem 1rem 2rem; }
    .hero { font-size: 36px; letter-spacing: -1px; }
    .metric-card { min-height: 125px; }
}
</style>
    """,
    unsafe_allow_html=True,
)

DATA_CANDIDATES = [
    PROJECT_DIR / "house_price_regression_dataset.csv",
    PROJECT_DIR / "data" / "house_price_regression_dataset.csv",
    APP_DIR / "house_price_regression_dataset.csv",
    APP_DIR / "data" / "house_price_regression_dataset.csv",
]

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_dataset(path_string):
    return pd.read_csv(path_string)

def section_heading(title, subtitle):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="gold-rule"></div>'
        f'<div class="desc">{subtitle}</div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

def chart_theme(fig):
    fig.update_layout(
        paper_bgcolor=CHARCOAL,
        plot_bgcolor=CHARCOAL,
        font=dict(family="DM Sans", color=TEXT),
        title_font=dict(family="Manrope", size=18, color=TEXT),
        margin=dict(l=20, r=20, t=48, b=25),
        legend=dict(bgcolor=CHARCOAL, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, color=MUTED, linecolor=GOLD)
    fig.update_yaxes(gridcolor="rgba(47,191,133,.18)", color=MUTED)
    return fig

# Load model and dataset independently so the dashboard can still open if either is absent.
model, model_error = None, None
try:
    model = get_model()
except Exception as exc:
    model_error = str(exc)

dataset, data_error = None, None
data_path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
if data_path:
    try:
        dataset = get_dataset(str(data_path))
    except Exception as exc:
        data_error = f"Dataset could not be read: {exc}"
else:
    data_error = "Dataset CSV not found. Put it in the project root or app/data folder."

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# SIDEBAR
with st.sidebar:
    st.markdown(
        '<div class="brand">HomeValue AI</div>'
        '<div class="brand-caption">PROPERTY INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Pages",
        ["Overview", "Predict House Price", "Model Performance", "Data Insights", "About Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="eyebrow">POWERED BY</div><h3>Machine Learning</h3><p>Linear Regression Pipeline</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Foundations of Practical Data Science")
    

# OVERVIEW
if page == "Overview":
    st.markdown(
        '<div class="eyebrow">PROPERTY INTELLIGENCE PLATFORM</div>'
        '<div class="hero">Understand your property.<br>Discover its estimated value.</div>'
        '<div class="desc">An interactive machine learning application that estimates residential property prices from seven property characteristics.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(4, gap="medium")
    cards = [
        ("MODEL", "Linear Regression", "Selected algorithm"),
        ("INPUT FEATURES", "07", "Property characteristics"),
        ("R² SCORE", f"{METRICS['R2']:.5f}", "Held-out test set"),
        ("PREDICTIONS", str(len(st.session_state.prediction_history)), "Current session"),
    ]
    for column, values in zip(columns, cards):
        with column:
            metric_card(*values)

    st.write("")
    left, right = st.columns([1.12, 1], gap="large", vertical_alignment="top")
    with left:
        section_heading("A smarter way to estimate property value", "An end-to-end machine learning experience.")
        st.markdown(
            '<div class="content-card"><h3>From property details to price estimation</h3>'
            '<p>Enter seven property characteristics and let the trained model generate an estimated house price.</p>'
            '<div class="gold-rule"></div><b>THE PREDICTION WORKFLOW</b>'
            '<p>01 &nbsp; Property information<br>02 &nbsp; Feature preparation<br>'
            '03 &nbsp; Trained ML pipeline<br>04 &nbsp; Estimated property price</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        section_heading("Model snapshot", "Performance recorded in the original regression assignment.")
        st.markdown(
            f'<div class="content-card"><div class="micro-label">MEAN ABSOLUTE ERROR</div>'
            f'<div class="metric-value">{METRICS["MAE"]:,.2f}</div><hr>'
            f'<div class="micro-label">ROOT MEAN SQUARED ERROR</div>'
            f'<div class="metric-value">{METRICS["RMSE"]:,.2f}</div><hr>'
            f'<div class="micro-label">R² SCORE</div>'
            f'<div class="metric-value">{METRICS["R2"]:.5f}</div></div>',
            unsafe_allow_html=True,
        )

    if model_error:
        st.warning("The dashboard is available, but the trained model is not ready yet.")
        with st.expander("Model setup details"):
            st.code(model_error)
            st.write("Checked model locations:")
            st.code("\n".join(str(p) for p in MODEL_SEARCH_PATHS))
            st.info("Run `python train_model.py` from this app folder after confirming the dataset is present.")
    if data_error:
        st.warning(data_error)

# PREDICTION
elif page == "Predict House Price":
    section_heading("Estimate your property's value", "Enter property characteristics to generate a prediction.")
    if model is None:
        st.error("The trained model is not available yet. Generate it by running `python train_model.py` from the application folder.")
        with st.expander("Model locations checked"):
            st.code("\n".join(str(p) for p in MODEL_SEARCH_PATHS))
        st.stop()
    if dataset is None:
        st.error(data_error)
        st.stop()
    missing = [c for c in FEATURE_COLUMNS if c not in dataset.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    medians = dataset[FEATURE_COLUMNS].median(numeric_only=True)
    st.markdown('<div class="content-card"><b>PROPERTY INFORMATION</b><p>Suggested starting values are based on the dataset medians. Enter values using the same units and scales as the training dataset.</p></div>', unsafe_allow_html=True)
    with st.form("property_prediction_form"):
        st.markdown("### Property specifications")
        first, second = st.columns(2, gap="large")
        with first:
            sqft = st.number_input("Square Footage", min_value=0.01, value=max(float(medians["Square_Footage"]), 0.01), step=50.0)
            bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=max(0, int(round(medians["Num_Bedrooms"]))), step=1)
            bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=max(0.0, float(medians["Num_Bathrooms"])), step=0.5)
            year = st.number_input("Year Built", min_value=1800, max_value=2100, value=int(round(medians["Year_Built"])), step=1)
        with second:
            lot = st.number_input("Lot Size", min_value=0.0, value=max(0.0, float(medians["Lot_Size"])), step=50.0)
            garage = st.number_input("Garage Size", min_value=0.0, value=max(0.0, float(medians["Garage_Size"])), step=1.0)
            quality = st.number_input("Neighborhood Quality", min_value=0.0, value=max(0.0, float(medians["Neighborhood_Quality"])), step=1.0)
        submitted = st.form_submit_button("ESTIMATE HOUSE PRICE", use_container_width=True)

    if submitted:
        details = {
            "Square_Footage": sqft,
            "Num_Bedrooms": bedrooms,
            "Num_Bathrooms": bathrooms,
            "Year_Built": year,
            "Lot_Size": lot,
            "Garage_Size": garage,
            "Neighborhood_Quality": quality,
        }
        try:
            price = predict_house_price(model, details)
            st.session_state.prediction_history.append({"Prediction": len(st.session_state.prediction_history) + 1, **details, "Predicted Price": price})
            st.markdown(
                f'<div class="result-card"><div class="micro-label">YOUR ESTIMATED PROPERTY VALUE</div>'
                f'<div class="result-price">{price:,.2f}</div>'
                '<p>Generated by your saved Linear Regression pipeline.</p></div>',
                unsafe_allow_html=True,
            )
            with st.expander("View property details"):
                st.json(details)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# MODEL PERFORMANCE
elif page == "Model Performance":
    section_heading("Model performance", "Evaluation results recorded in the original regression notebook.")
    columns = st.columns(4, gap="medium")
    for column, values in zip(columns, [
        ("MAE", f"{METRICS['MAE']:,.2f}", "Mean Absolute Error"),
        ("MSE", f"{METRICS['MSE']:,.2f}", "Mean Squared Error"),
        ("RMSE", f"{METRICS['RMSE']:,.2f}", "Root Mean Squared Error"),
        ("R² SCORE", f"{METRICS['R2']:.5f}", "Held-out test set"),
    ]):
        with column:
            metric_card(*values)
    st.write("")
    section_heading("Cross-validation comparison", "Five-fold cross-validation results from the notebook.")
    fig = px.bar(CV_RESULTS, x="Model", y="Mean CV R²", color_discrete_sequence=[GOLD], text="Mean CV R²")
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.update_yaxes(range=[0.985, 1.002], title="Mean CV R²")
    st.plotly_chart(chart_theme(fig), use_container_width=True)
    st.dataframe(CV_RESULTS, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="content-card"><h3>Selected model: Linear Regression</h3>'
        '<p>Reported mean five-fold cross-validation R²: 0.998500. The saved model is built as a Scikit-learn Pipeline containing StandardScaler and LinearRegression.</p>'
        '<p>Displayed metrics are from the original notebook; this page does not recalculate them.</p></div>',
        unsafe_allow_html=True,
    )

# DATA INSIGHTS
elif page == "Data Insights":
    section_heading("Explore the dataset", "Discover patterns and relationships in the original house price dataset.")
    if dataset is None:
        st.error(data_error)
        st.stop()
    if "House_Price" not in dataset.columns:
        st.error("House_Price target column not found.")
        st.stop()
    columns = st.columns(3, gap="medium")
    for column, values in zip(columns, [
        ("TOTAL RECORDS", f"{len(dataset):,}", "Dataset observations"),
        ("TOTAL COLUMNS", str(len(dataset.columns)), "Features and target"),
        ("MISSING VALUES", f"{dataset.isna().sum().sum():,}", "Across the dataset"),
    ]):
        with column:
            metric_card(*values)
    st.write("")
    section_heading("Dataset preview", "First 15 records from the source CSV.")
    st.dataframe(dataset.head(15), use_container_width=True, hide_index=True)
    left, right = st.columns(2, gap="large")
    with left:
        section_heading("House price distribution", "Distribution of the target variable.")
        st.plotly_chart(chart_theme(px.histogram(dataset, x="House_Price", nbins=30, color_discrete_sequence=[GOLD])), use_container_width=True)
    with right:
        section_heading("Square footage analysis", "Relationship between property size and price.")
        st.plotly_chart(chart_theme(px.scatter(dataset, x="Square_Footage", y="House_Price", color_discrete_sequence=[GOLD])), use_container_width=True)
    section_heading("Feature correlation", "Correlation between numerical dataset columns.")
    correlation = dataset.select_dtypes(include="number").corr()
    heatmap = px.imshow(correlation, text_auto=".2f", color_continuous_scale=[CHARCOAL, GOLD], aspect="auto")
    st.plotly_chart(chart_theme(heatmap), use_container_width=True)
    section_heading("Descriptive statistics", "Statistical summary of numerical columns.")
    st.dataframe(dataset.describe().T, use_container_width=True)

# ABOUT
elif page == "About Project":
    st.markdown(
        '<div class="eyebrow">ABOUT THE PROJECT</div>'
        '<div class="hero">Intelligence behind<br>every estimate.</div>'
        '<div class="desc">HomeValue AI demonstrates how practical data science and machine learning can transform structured property data into estimated house prices.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    section_heading("Machine learning workflow", "From raw data to an interactive prediction experience.")
    workflow = [
        ("01", "Data Collection", "House price regression dataset"),
        ("02", "Data Exploration", "Understanding features and relationships"),
        ("03", "Preprocessing", "StandardScaler and feature preparation"),
        ("04", "Model Training", "Regression algorithm experimentation"),
        ("05", "Evaluation", "MAE, MSE, RMSE, R² and cross-validation"),
        ("06", "Deployment", "Joblib, Streamlit and Docker"),
    ]
    for number, title, description in workflow:
        st.markdown(f'<div class="content-card" style="margin-bottom:12px"><b>{number} &nbsp; {title}</b><p>{description}</p></div>', unsafe_allow_html=True)
    section_heading("Technology stack", "Tools used to develop the project.")
    st.markdown('<div class="content-card"><b>Programming:</b> Python<br><br><b>Data processing:</b> Pandas, NumPy<br><br><b>Machine learning:</b> Scikit-learn<br><br><b>Interface:</b> Streamlit<br><br><b>Visualisation:</b> Plotly<br><br><b>Model persistence:</b> Joblib<br><br><b>Deployment:</b> Docker</div>', unsafe_allow_html=True)
    st.write("")
    section_heading("Project information", "Academic project details.")
    st.markdown('<div class="content-card"><h3>HomeValue AI — House Price Prediction System</h3><p><b>Student:</b> Rohit<br><b>Program:</b> BCA (AI & DS) – A<br><b>Year:</b> Second Year<br><b>Course:</b> Foundations of Practical Data Science<br><b>Course code:</b> ETCADA303</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:22px 0">'
    '<div class="brand">HomeValue AI</div>'
    '<div class="brand-caption">PROPERTY INTELLIGENCE · POWERED BY MACHINE LEARNING</div>'
    '<p style="margin-top:12px;opacity:.7">Foundations of Practical Data Science</p></div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style="text-align:center; padding:20px; color:#888;">
        © 2026 HomeValue AI. All Rights Reserved.
        <br>
        Created by Rohit.
    </div>
    """,
    unsafe_allow_html=True
)