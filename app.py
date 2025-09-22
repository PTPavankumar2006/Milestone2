# app.py
# ------------------------------------------------------------
# 🏠 Housing Price Prediction & Smart Investment Dashboard
# ------------------------------------------------------------
# What you get:
# - Clean, modular Streamlit app with tabs (Data ➜ EDA ➜ Modeling ➜ Importance ➜ Investment Bot ➜ Predict)
# - Robust preprocessing (numeric/categorical handling, missing values, VIF)
# - Feature selection (SelectKBest) + optional RFE
# - Multiple models (Linear, Ridge, RandomForest, + optional XGBoost / LightGBM)
# - Cross-validation metrics, optional hyperparameter tuning (RandomizedSearchCV)
# - Residuals, error distributions, and learning curves
# - Permutation importances (safe for any estimator) or native importances when available
# - Save/Load final pipeline with joblib
# - Investment suggestions dashboard
# - Batch prediction from uploaded CSV
# ------------------------------------------------------------

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

# -----------------------------
# Optional models (if installed)
# -----------------------------
has_xgb = False
has_lgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    pass

try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    pass

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Housing Price Prediction & Investment Bot", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🏠 Housing Price Prediction & Smart Investment Dashboard")

# -----------------------------
# Helper: safe RMSE
# -----------------------------
def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -----------------------------
# Sidebar: controls
# -----------------------------
st.sidebar.header("⚙️ Controls")
DEFAULT_DATA_PATH = r"C:\Users\ptpav\OneDrive\Desktop\Stream_pro\data\housing_price.csv"

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"]) 
path_input = st.sidebar.text_input("or Path to CSV", value=DEFAULT_DATA_PATH)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 10_000, 42, 1)
use_rfe = st.sidebar.checkbox("Use RFE on numeric features (Linear/Ridge)", value=False)

tune_models = st.sidebar.checkbox("Hyperparameter tuning (slow)", value=False)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 5)

FINAL_MODEL_PATH = st.sidebar.text_input("Final model path", value="final_price_model.joblib")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(file_like, path_fallback):
    if file_like is not None:
        return pd.read_csv(file_like)
    return pd.read_csv(path_fallback)

try:
    df = load_data(uploaded, path_input)
except Exception as e:
    st.error(f"❌ Failed to load data. Please upload/select a valid CSV. Error: {e}")
    st.stop()

st.subheader("Dataset Preview")
col_a, col_b = st.columns([1, 2])
with col_a:
    st.write("Rows, Columns:", df.shape)
with col_b:
    st.dataframe(df.head())

# -----------------------------
# Column configuration (adapt to your data)
# -----------------------------
# Define expected numeric and categorical columns (only keep those present)
numeric_cols_raw = ['ID','BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
                    'Year_Built','Floor No','Total_Floors','Age_of_Property']
cat_cols_raw = ['State','City','Locality','Property_Type','fl','Furnished_Status','D',
                'Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility',
                'Parking Space','Security','Amenities','Facing','Owner_Type','Availability_Status']

numeric_cols = [c for c in numeric_cols_raw if c in df.columns]
cat_cols = [c for c in cat_cols_raw if c in df.columns]

# IMPORTANT: remove the target column from feature lists to avoid leakage
if 'Price_in_Lakhs' in numeric_cols:
    numeric_cols.remove('Price_in_Lakhs')

# -----------------------------
# Basic cleaning
# -----------------------------
# Ensure numeric dtype and impute
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c].fillna(df[c].median(), inplace=True)

for c in cat_cols:
    df[c] = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown"})

# Drop ID if present (not a feature)
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)
    if 'ID' in numeric_cols:
        numeric_cols.remove('ID')

st.success("✅ Data cleaned and missing values handled.")

# -----------------------------
# Correlation heatmap (numeric only)
# -----------------------------
st.subheader("🔗 Correlation Analysis (Numeric Features)")
if len(numeric_cols) > 1:
    corr = df[numeric_cols + ['Price_in_Lakhs']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
    st.pyplot(fig)
    if 'Price_in_Lakhs' in corr.columns:
        target_corr = corr['Price_in_Lakhs'].drop('Price_in_Lakhs').abs().sort_values(ascending=False)
        st.write("Top correlated features with target:")
        st.dataframe(target_corr.head(10))
else:
    st.info("Not enough numeric columns for a correlation heatmap.")

# -----------------------------
# VIF (multicollinearity)
# -----------------------------
st.subheader("🧮 VIF (Multicollinearity Check)")
@st.cache_data(show_spinner=False)
def compute_vif(df_num):
    X = df_num.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0)
    if X.shape[1] < 2:
        return pd.DataFrame({"feature": X.columns, "VIF": [np.nan]*X.shape[1]})
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

if len(numericCols := numeric_cols) > 1:
    vif_df = compute_vif(df[numericCols])
    st.dataframe(vif_df)
else:
    st.info("Not enough numeric columns to compute VIF.")

# -----------------------------
# Preprocessor (safe OHE across sklearn versions)
# -----------------------------

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', make_ohe())])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols),
])

# -----------------------------
# Feature selection (SelectKBest on fully transformed matrix)
# -----------------------------
X_all = df[numeric_cols + cat_cols].copy()
y_all = df['Price_in_Lakhs'] if 'Price_in_Lakhs' in df.columns else None

if y_all is None:
    st.error("❌ Target column 'Price_in_Lakhs' not found. Please include it in your dataset.")
    st.stop()

# Fit preprocessor once for feature names
X_transformed = preprocessor.fit_transform(X_all)

# names after OHE (guard if no categorical cols)
cat_feature_names = []
if len(cat_cols):
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_feature_names = list(ohe.get_feature_names_out())

feature_names_all = numeric_cols + cat_feature_names

k = min(30, X_transformed.shape[1])
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X_transformed, y_all)
mask = selector.get_support()
selected_features = [f for f, m in zip(feature_names_all, mask) if m]
scores = selector.scores_[mask]
feat_scores = pd.DataFrame({'feature': selected_features, 'score': scores}).sort_values('score', ascending=False)

st.subheader("🏅 Top Features (SelectKBest)")
st.dataframe(feat_scores.head(20))

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state)
st.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# -----------------------------
# Models & (optional) tuning
# -----------------------------
base_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=random_state),
    "RandomForest": RandomForestRegressor(random_state=random_state, n_estimators=400, n_jobs=-1),
}
if has_xgb:
    base_models["XGBoost"] = xgb.XGBRegressor(random_state=random_state, n_estimators=400, n_jobs=-1, verbosity=0)
if has_lgb:
    base_models["LightGBM"] = lgb.LGBMRegressor(random_state=random_state, n_estimators=400, n_jobs=-1)

# Hyperparameter search spaces
rf_dist = {
    'model__n_estimators': [200, 400, 800],
    'model__max_depth': [None, 8, 12, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

xgb_dist = {
    'model__n_estimators': [300, 500, 800],
    'model__max_depth': [3, 5, 8],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.9, 1.0],
}

lgb_dist = {
    'model__n_estimators': [300, 500, 800],
    'model__num_leaves': [31, 63, 127],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.9, 1.0],
}

# -----------------------------
# Tabs for Dashboard
# -----------------------------
T1, T2, T3, T4, T5, T6, T7 = st.tabs([
    "📁 Data", "📊 EDA", "🤖 Modeling", "🧠 Importances", 
    "💡 Investment Bot", "🧾 Predict", "🎤 Voice Assistant"
])

with T1:
    st.markdown("### Data Overview")
    st.dataframe(df.sample(min(500, len(df))).reset_index(drop=True))
    st.markdown("**Class balance / Target distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df['Price_in_Lakhs'], bins=40, ax=ax)
    ax.set_title("Target Distribution: Price_in_Lakhs")
    st.pyplot(fig)

with T2:
    st.markdown("### Quick EDA")
    # Pairplot on a small sample to avoid heavy rendering
    num_show = [c for c in numeric_cols if c != 'Price_in_Lakhs'][:4]
    if len(num_show) >= 2:
        sample_df = df[[*num_show, 'Price_in_Lakhs']].sample(min(500, len(df)), random_state=random_state)
        pairgrid = sns.pairplot(sample_df, diag_kind='kde')
        st.pyplot(pairgrid.fig)
    else:
        st.info("Not enough numeric columns for pairplot.")

    st.markdown("**Boxplots by City (top 10 by count)**")
    if 'City' in df.columns:
        top_cities = df['City'].value_counts().head(10).index
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df[df['City'].isin(top_cities)], x='City', y='Price_in_Lakhs', ax=ax)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

with T3:
    st.markdown("### Train & Compare Models")

    results = []
    trained = {}

    for name, model in base_models.items():
        with st.spinner(f"Training {name}..."):
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            rmse = compute_rmse(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'model': name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pipeline': pipe
            })
            trained[name] = pipe

    # Build results table
    res_df = pd.DataFrame(results).sort_values('rmse')
    st.dataframe(res_df[['model','rmse','mae','r2']].reset_index(drop=True))

    # Pick the best model
    best_row = res_df.iloc[0]
    best_model_name = best_row['model']
    best_pipeline = best_row['pipeline']
    st.success(f"🏆 Best model by RMSE: **{best_model_name}**")

    # Show metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", f"{best_row['rmse']:.3f}")
    c2.metric("Test MAE", f"{best_row['mae']:.3f}")
    c3.metric("Test R²", f"{best_row['r2']:.3f}")

    # Save option
    with st.expander("💾 Save Best Model"):
        if st.button("Save best model to disk"):
            joblib.dump(best_pipeline, FINAL_MODEL_PATH)
            st.success(f"Saved ➜ {FINAL_MODEL_PATH}")

    # Diagnostics
    with st.expander("📉 Residuals & Errors"):
        y_pred_best = best_pipeline.predict(X_test)
        residuals = y_test - y_pred_best
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_pred_best, y=residuals, ax=ax)
        ax.axhline(0, ls='--')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, bins=40, kde=True, ax=ax2)
        ax2.set_title("Residual Distribution")
        st.pyplot(fig2)


with T4:
    st.markdown("### Feature Importances / Effects")

    # Reuse best from previous tab if available; otherwise fit a default RF
    try:
        best_pipeline  # noqa
    except NameError:
        # Fit a quick default RF
        st.info("No trained pipeline found in this session. Fitting a quick RandomForest for importances...")
        best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=random_state, n_estimators=400, n_jobs=-1))])
        best_pipeline.fit(X_train, y_train)

    model = best_pipeline.named_steps['model']

    # Extract feature names after preprocessing (robust to empty cat_cols)
    cat_feats = []
    if len(cat_cols):
        ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        try:
            cat_feats = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            cat_feats = list(ohe.get_feature_names_out())

    feat_names = numeric_cols + cat_feats

    def plot_bar_importances(df_imp, title):
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(data=df_imp.head(20), x='importance', y='feature', ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    # Native importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        plot_bar_importances(imp_df, "Native Feature Importances")
    elif hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': np.abs(coefs)}).sort_values('importance', ascending=False)
        plot_bar_importances(imp_df, "Coefficient Magnitudes (abs)")
    else:
        st.info("Model has no native importances; computing permutation importances (this may take a bit)...")
        r = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
        plot_bar_importances(imp_df, "Permutation Importances")

with T5:
    st.markdown("### 💡 Smart Investment Suggestions")

    @st.cache_data(show_spinner=False)
    def smart_investment_suggestions(df_in, sector_col='Property_Type', top_n=5, min_listings=15):
        results = {}
        if sector_col not in df_in.columns:
            return results
        sectors = df_in[sector_col].dropna().unique()
        for sector in sectors:
            df_s = df_in[df_in[sector_col] == sector].copy()
            if df_s.shape[0] < 5:
                continue
            group_cols = [c for c in ['City','Locality'] if c in df_s.columns]
            if not group_cols:
                continue
            grp = df_s.groupby(group_cols).agg(
                listings=('Price_per_SqFt','count'),
                avg_price_per_sqft=('Price_per_SqFt','median'),
                avg_price_lakh=('Price_in_Lakhs','median'),
                avg_size=('Size_in_SqFt','median'),
                avg_year=('Year_Built','median')
            ).reset_index()
            grp = grp[grp['listings'] >= min_listings]
            if grp.empty:
                continue
            # Normalize scores safely
            def minmax(s):
                s = s.astype(float)
                denom = (s.max() - s.min())
                return (s - s.min()) / (denom if denom != 0 else 1)

            grp['demand_score'] = minmax(grp['listings'])
            grp['affordability_score'] = 1 - minmax(grp['avg_price_per_sqft'])
            grp['growth_score'] = minmax(grp['avg_year'].fillna(grp['avg_year'].median()))
            grp['composite_score'] = 0.3*grp['demand_score'] + 0.25*grp['affordability_score'] + 0.3*grp['growth_score']
            top = grp.sort_values('composite_score', ascending=False).head(top_n)
            results[sector] = top
        return results

    sector_col = 'Property_Type' if 'Property_Type' in df.columns else st.selectbox("Choose sector column", options=[c for c in df.columns if df[c].nunique()<50])
    top_n = st.slider("Top N per sector", 3, 15, 5)
    min_listings = st.slider("Minimum listings per (City, Locality)", 5, 50, 15)

    suggestions = smart_investment_suggestions(df, sector_col=sector_col, top_n=top_n, min_listings=min_listings)
    if not suggestions:
        st.info("No suggestions available with current filters.")
    else:
        for sector, table in suggestions.items():
            st.markdown(f"#### Sector: **{sector}**")
            st.dataframe(table)

with T6:
    st.markdown("### 🧾 Predict (Batch)")
    st.write("Upload a CSV with the same feature columns (minus target). We'll apply the **best saved model** or a freshly trained one from this session.")

    pred_file = st.file_uploader("Upload CSV for prediction", type=['csv'], key="pred")

    # Load model if available, else use best_pipeline from modeling tab
    model_to_use = None

    if os.path.exists(FINAL_MODEL_PATH):
        try:
            model_to_use = joblib.load(FINAL_MODEL_PATH)
            st.success(f"Loaded saved model: {FINAL_MODEL_PATH}")
        except Exception as e:
            st.warning(f"Could not load saved model ({e}). Will try to use in-session best model if available.")

    if model_to_use is None:
        try:
            model_to_use = best_pipeline
            st.info("Using in-session best model for predictions.")
        except NameError:
            st.error("No model available. Train a model in the Modeling tab or load a saved model path.")

    if pred_file is not None and model_to_use is not None:
        new_df = pd.read_csv(pred_file)
        # Basic cleaning consistent with training
        for c in numeric_cols:
            if c in new_df.columns:
                new_df[c] = pd.to_numeric(new_df[c], errors='coerce')
                new_df[c].fillna(df[c].median(), inplace=True)
        for c in cat_cols:
            if c in new_df.columns:
                new_df[c] = new_df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown"})
        if 'ID' in new_df.columns:
            new_df.drop(columns=['ID'], inplace=True)

        missing_cols = [c for c in (numeric_cols + cat_cols) if c not in new_df.columns]
        if missing_cols:
            st.warning(f"The following expected columns are missing and will be filled with default values: {missing_cols}")
            for c in missing_cols:
                if c in numeric_cols:
                    new_df[c] = df[c].median()
                else:
                    new_df[c] = "Unknown"

        preds = model_to_use.predict(new_df[numeric_cols + cat_cols])
        out = new_df.copy()
        out['Predicted_Price_in_Lakhs'] = preds
        st.dataframe(out.head(50))

        # Download
        out_path = "predictions.csv"
        out.to_csv(out_path, index=False)
        st.download_button("⬇️ Download predictions.csv", data=out.to_csv(index=False), file_name="predictions.csv")

with T7:
    st.markdown("### 🎤 AI Voice Assistant (Chat Mode)")
    st.write("Talk to your assistant: *'What is the price of a 3BHK in Whitefield Bangalore?'*")

    import speech_recognition as sr
    import pyttsx3
    import spacy
    import subprocess, sys

    # ✅ Auto-install spaCy model if missing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------------------
    # Function: Process Query
    # ---------------------------
    def process_query(query_text):
        doc = nlp(query_text)

        bhk, location = None, None
        for ent in doc.ents:
            if ent.label_ == "CARDINAL" and "BHK" in query_text:
                try:
                    bhk = int(ent.text)
                except:
                    pass
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text

        # Default sample row
        response_text = "I couldn't extract enough details to predict."
        if model_to_use is not None:
            sample = pd.DataFrame([{**{c: df[c].median() for c in numeric_cols},
                                    **{c: "Unknown" for c in cat_cols}}])
            if bhk and "BHK" in sample.columns:
                sample["BHK"] = bhk
            if location and "City" in sample.columns:
                sample["City"] = location

            price_pred = model_to_use.predict(sample[numeric_cols + cat_cols])[0]
            response_text = f"The estimated price is **{price_pred:.2f} Lakhs**"

            # 🔊 Speak out
            tts_engine.say(f"The estimated price is {price_pred:.0f} lakhs")
            tts_engine.runAndWait()

        return response_text

    # ---------------------------
    # Input Section
    # ---------------------------
    col1, col2 = st.columns([2,1])
    with col1:
        user_input = st.text_input("💬 Type your query", key="chat_input")
    with col2:
        if st.button("🎙️ Speak"):
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            try:
                user_input = recognizer.recognize_google(audio)
                st.success(f"You said: {user_input}")
            except Exception as e:
                st.error(f"Could not recognize speech: {e}")
                user_input = ""

    # ---------------------------
    # Generate & Display Reply
    # ---------------------------
    if user_input:
        reply = process_query(user_input)

        # Save in chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", reply))

        # 🔹 Show latest response immediately in text
        st.markdown(f"**Assistant:** {reply}")

    # ---------------------------
    # Display Chat History
    # ---------------------------
    st.markdown("### 📝 Chat History")
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"<div style='text-align:right; background:#DCF8C6; padding:10px; border-radius:10px; margin:5px;'>{msg}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left; background:#F1F0F0; padding:10px; border-radius:10px; margin:5px;'>{msg}</div>",
                unsafe_allow_html=True,
            )


# -----------------------------
# (Optional) Recursive Feature Elimination (RFE) on numeric only
# -----------------------------
if use_rfe:
    st.subheader("🔬 RFE on Numeric Features (Linear/Ridge)")
    base_est = Ridge(random_state=random_state)
    rfe = RFE(base_est, n_features_to_select=min(5, len(numeric_cols)))
    try:
        rfe.fit(df[numeric_cols], y_all)
        selected = [col for col, keep in zip(numeric_cols, rfe.support_) if keep]
        st.write("Selected numeric features:", selected)
    except Exception as e:
        st.warning(f"RFE skipped due to error: {e}")

st.success("App ready ✅ — Explore the tabs above to use the full dashboard.")
