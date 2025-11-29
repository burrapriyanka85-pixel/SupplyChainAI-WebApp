# app.py -- SupplyChainAI web app (single-file)
# Requirements: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sns.set_theme(style="darkgrid")

# Page config
st.set_page_config(
    page_title="AI-Powered Supply Chain Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Generic helpers ----------------
def load_csv_try(fileobj) -> pd.DataFrame:
    """Try reading uploaded csv, fallback attempts for encodings."""
    try:
        fileobj.seek(0)
        return pd.read_csv(fileobj)
    except Exception:
        try:
            fileobj.seek(0)
            return pd.read_csv(fileobj, encoding="latin1")
        except Exception:
            try:
                fileobj.seek(0)
                raw = fileobj.read()
                if isinstance(raw, bytes):
                    return pd.read_csv(io.StringIO(raw.decode("utf8", errors="ignore")))
            except Exception:
                pass
            raise

def show_basic_stats(df):
    st.write("### Dataset snapshot")
    st.dataframe(df.head(10))
    st.write("### Basic info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Missing values by column:")
    st.dataframe(df.isna().sum().sort_values(ascending=False).head(30))

def drop_high_cardinality_and_useless(df, cat_card_threshold=1000, extra_drop=None):
    if extra_drop is None:
        extra_drop = []
    df = df.copy()
    drop_cols = []
    for col in df.columns:
        low = col.lower() if isinstance(col, str) else ""
        if ("email" in low) or ("password" in low) or ("image" in low) or ("img" in low) or ("url" in low):
            drop_cols.append(col)
        if low.endswith("id") or low.endswith("_id") or low == "id" or "order id" in low:
            drop_cols.append(col)
    drop_cols = list(set(drop_cols + extra_drop))
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            if df[col].nunique(dropna=True) > cat_card_threshold:
                drop_cols.append(col)
        except Exception:
            pass
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        st.info(f"Dropping columns (useless / very high-card): {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df, drop_cols

def build_preprocessing_pipeline(X: pd.DataFrame, ordinal_unknown_value=-1):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=ordinal_unknown_value))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols

def train_model(X, y, preprocessor, rf_params=None):
    if rf_params is None:
        rf_params = {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1}
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**rf_params))
    ])
    model.fit(X, y)
    try:
        model.input_columns = list(X.columns)
        model.numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        model.categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    except Exception:
        pass
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    unique_classes = np.unique(y_test)
    if len(unique_classes) <= 2:
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
    else:
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "preds": preds}

def plot_confusion_matrix(cm, labels=None):
    if labels is None:
        labels = np.arange(cm.shape[0])
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def get_feature_importances(model, numeric_cols, categorical_cols, top_n=20):
    try:
        clf = model.named_steps["classifier"]
        importances = clf.feature_importances_
        feature_names = list(numeric_cols) + list(categorical_cols)
        if len(importances) != len(feature_names):
            try:
                pre = model.named_steps["preprocessor"]
                names = []
                if hasattr(pre, "transformers_"):
                    for name, trans, cols in pre.transformers_:
                        if name == "num":
                            names.extend(cols)
                        elif name == "cat":
                            names.extend(cols)
                feature_names = names
            except Exception:
                pass
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return fi.head(top_n)
    except Exception:
        return None

def infer_model_input_columns(model):
    """Try to recover the input column names the model expects."""
    try:
        if hasattr(model, "input_columns") and getattr(model, "input_columns"):
            inp = list(getattr(model, "input_columns"))
            return inp, list(getattr(model, "numeric_cols", []) or []), list(getattr(model, "categorical_cols", []) or [])
    except Exception:
        pass

    pre = None
    numeric_cols = []
    cat_cols = []
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
    except Exception:
        pre = None

    if pre is not None:
        if hasattr(pre, "transformers_"):
            for name, transformer, cols in pre.transformers_:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    if name == "num":
                        numeric_cols.extend(list(cols))
                    elif name == "cat":
                        cat_cols.extend(list(cols))
            expected = list(cat_cols) + list(numeric_cols)
            if expected:
                return expected, numeric_cols, cat_cols

    try:
        final = model
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            final = model.named_steps["classifier"]
        if hasattr(final, "feature_names_in_"):
            cols = list(final.feature_names_in_)
            return cols, [], []
    except Exception:
        pass

    return None, numeric_cols, cat_cols

def autofill_missing_columns_for_model(df, expected_cols, numeric_cols):
    """Auto-fill missing columns: numeric -> 0, categorical -> 'UNKNOWN'"""
    df = df.copy()
    added = []
    for col in expected_cols:
        if col not in df.columns:
            if col in numeric_cols:
                df[col] = 0
            else:
                df[col] = "UNKNOWN"
            added.append(col)
    return df, added

def safe_map_target_using_model(y_series, model):
    """Map a target series (possibly strings) into numeric labels expected by the model."""
    if pd.api.types.is_numeric_dtype(y_series):
        return y_series, None

    mapping = getattr(model, "label_mapping", None)
    if mapping:
        try:
            mapped = y_series.map(mapping)
            return mapped, mapping
        except Exception:
            pass

    uniques = list(y_series.dropna().unique())
    if not uniques:
        return y_series, None

    pos = uniques[0]
    maph = {pos: 1}
    for u in uniques:
        if u != pos:
            maph[u] = 0
    mapped = y_series.map(maph)
    return mapped, maph

def prepare_df_for_model(df, model, numeric_defaults=0, cat_default="UNKNOWN"):
    """Ensure `df` has the exact columns the model expects (order too)."""
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    expected, numeric_cols_in_model, cat_cols_in_model = infer_model_input_columns(model)
    if expected is None:
        expected = ["Distance", "Freight Cost", "Weight"]

    csv_cols_lower = {c.lower(): c for c in df.columns if isinstance(c, str)}
    rename_map = {}
    for exp in expected:
        if not isinstance(exp, str):
            continue
        if exp in df.columns:
            continue
        exp_l = exp.lower()
        if exp_l in csv_cols_lower:
            rename_map[csv_cols_lower[exp_l]] = exp
        else:
            matches = difflib.get_close_matches(exp_l, list(csv_cols_lower.keys()), n=1, cutoff=0.72)
            if matches:
                rename_map[csv_cols_lower[matches[0]]] = exp
    if rename_map:
        df = df.rename(columns=rename_map)

    df_reindexed = df.reindex(columns=expected)

    heur_numeric_keywords = ("cost", "distance", "weight", "qty", "quantity", "price", "amount")
    for col in df_reindexed.columns:
        if df_reindexed[col].isna().all():
            if any(k in str(col).lower() for k in heur_numeric_keywords):
                df_reindexed[col] = numeric_defaults
            else:
                df_reindexed[col] = cat_default

    for k in ("Distance", "Freight Cost", "Weight"):
        if k in df_reindexed.columns:
            df_reindexed[k] = pd.to_numeric(df_reindexed[k], errors="coerce").fillna(numeric_defaults)

    for c in df_reindexed.columns:
        if df_reindexed[c].dtype == object:
            df_reindexed[c] = df_reindexed[c].apply(lambda x: "" if pd.isna(x) else str(x))

    added = [c for c in df_reindexed.columns if c not in df.columns]
    return df_reindexed, added

def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

# ---------------- Session storage ----------------
if "last_model" not in st.session_state:
    st.session_state["last_model"] = None
if "last_model_name" not in st.session_state:
    st.session_state["last_model_name"] = None
if "menu" not in st.session_state:
    st.session_state["menu"] = "Home"

# ---------------- UI: Sidebar ----------------
st.sidebar.title("Choose page")
menu = st.sidebar.selectbox(
    "Choose page",
    ["Home", "Upload & EDA", "Train Model", "Model Results", "Predict", "Export/Load Model", "About"],
    index=0
)
st.session_state.menu = menu

# ---------- Home (compact, centered hero to match second screenshot) ----------
# ---------- Home (ORIGINAL LAYOUT YOU WANT) ----------
if menu == "Home":

    st.markdown(
        """
        <style>
        .stApp {
            background: #0b0c0d;
            color: #d7d7d9;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
        }
        .hero-container { max-width: 1400px; margin: 0 auto; padding: 20px 30px; }
        .hero-title { font-size: 36px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
        .hero-sub { font-size: 20px; color: #d0d4da; margin-bottom: 14px; }
        .small-muted { color: #b0b7c3; font-size: 15px; }
        .metric-card { padding: 18px; background: rgba(255,255,255,0.03); border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-container">', unsafe_allow_html=True)

    # Hero section - two columns
    c1, c2 = st.columns([2.2, 1], gap="large")

    with c1:
        st.markdown('<div class="hero-title">AI-Powered Supply Chain Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Predict Delivery Delays <strong>Before They Happen</strong></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='small-muted'>A data-driven platform using machine learning to forecast supply chain delivery delays with high accuracy. Make informed decisions and optimize your logistics.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Get Started"):
                st.session_state.menu = "Model Results"
        with colB:
            st.markdown("<a href='#Model Results'>View Demo</a>", unsafe_allow_html=True)

        # Feature list just like screenshot
        st.markdown("---")
        st.markdown(
            """
            **Real-time Analytics** — Monitor delivery performance with interactive dashboards and KPI tracking.  
            **Accurate Predictions** — Advanced Random Forest models tuned for supply chain data.  
            **Batch Processing** — Upload CSV files to predict multiple deliveries at once.  
            **Model Insights** — Understand what factors drive delays using feature importances.  
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.image(
            "https://images.unsplash.com/photo-1556761175-129418cb2dfe?q=80&w=1400&auto=format&fit=crop",
            caption="Supply chain analytics",
            use_container_width=True,
        )

    # Metrics row
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown('<div class="metric-card"><h3>98.12%</h3>Accuracy</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h3>50K+</h3>Predictions</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>23</h3>Features</div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><h3><1s</h3>Response Time</div>', unsafe_allow_html=True)

    # Feature columns (bottom section)
    st.markdown("---")

    f1, f2 = st.columns(2)
    with f1:
        st.subheader("Real-time Analytics")
        st.write("Monitor delivery performance with dashboards and KPI tracking.")
        st.write("- Drill-down by region\n- Alerts & thresholds")
        st.subheader("Batch Processing")
        st.write("Predict multiple deliveries at once from uploaded CSV files.")

    with f2:
        st.subheader("Accurate Predictions")
        st.write("Random Forest model tuned on historical delivery data.")
        st.subheader("Model Insights")
        st.write("Understand delay drivers with explainable feature importances.")

    st.markdown("---")
    st.caption("© 2025 SupplyChainAI — A Data-Driven Model for Predicting Delivery Delays in Supply Chains")

# ---------- Upload & EDA ----------
if menu == "Upload & EDA":
    st.header("Upload dataset and explore")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_csv_try(uploaded_file)
            st.success("File loaded")
            show_basic_stats(df)
            st.write("---")
            st.subheader("Columns")
            st.write(df.columns.tolist())
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.exception(traceback.format_exc())

# ---------- Train Model ----------
if menu == "Train Model":
    st.header("Train Random Forest model")
    uploaded_file = st.file_uploader("Upload CSV for training", type=["csv"], key="train_upload")
    if uploaded_file is not None:
        try:
            raw_df = load_csv_try(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            raw_df = None

        if raw_df is not None:
            st.success("File loaded for training")
            st.write("Columns detected:", raw_df.columns.tolist())

            extra_drop_input = st.text_input("Comma-separated extra columns to drop (optional)", value="")
            extra_drop = [c.strip() for c in extra_drop_input.split(",") if c.strip()]

            use_sample = st.checkbox("Sample rows for quick prototyping (recommended if dataset is large)", value=True)
            sample_frac = st.slider("Sample fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01) if use_sample else 1.0

            try:
                df_clean, dropped_columns = drop_high_cardinality_and_useless(raw_df, cat_card_threshold=1000, extra_drop=extra_drop)
            except Exception:
                df_clean = raw_df.copy()
                dropped_columns = []

            if use_sample and sample_frac < 1.0:
                st.info(f"Sampling {sample_frac*100:.1f}% of rows for quick training")
                df_clean = df_clean.sample(frac=float(sample_frac), random_state=42)

            st.write("---")
            target_col = st.selectbox("Select target (binary or numeric) column", options=[None] + df_clean.columns.tolist())
            if target_col:
                if pd.api.types.is_numeric_dtype(df_clean[target_col]):
                    st.info("You selected a numeric target. You can convert it to a binary 'delayed' label (recommended).")
                    if "Days for shipment (scheduled)" in df_clean.columns:
                        convert_to_binary = st.checkbox("Convert numeric days -> delayed (real > scheduled)?", value=True)
                    else:
                        convert_to_binary = st.checkbox("Convert numeric days -> delayed? (scheduled column not found)", value=False)

                    if convert_to_binary:
                        if "Days for shipment (scheduled)" in df_clean.columns:
                            df_clean["__delay_binary__"] = (df_clean[target_col] > df_clean["Days for shipment (scheduled)"]).astype(int)
                            st.success("Created binary target '__delay_binary__' where 1 = delayed")
                            target_col_internal = "__delay_binary__"
                        else:
                            st.error("Scheduled days column 'Days for shipment (scheduled)' not found; cannot auto-convert.")
                            target_col_internal = target_col
                    else:
                        target_col_internal = target_col
                else:
                    target_col_internal = target_col

                y = df_clean[target_col_internal].copy()
                X = df_clean.drop(columns=[target_col_internal])

                if not pd.api.types.is_numeric_dtype(y):
                    vals = y.dropna().unique().tolist()
                    st.write("Detected non-numeric target values:", vals)
                    if vals:
                        pos = st.selectbox("Choose which value indicates DELAY (1)", options=vals, index=0)
                        mapping = {pos: 1}
                        for v in vals:
                            if v != pos:
                                mapping[v] = 0
                        y = y.map(mapping)

                mask = ~y.isna()
                X = X.loc[mask].reset_index(drop=True)
                y = y.loc[mask].reset_index(drop=True)

                test_size = st.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
                random_state = int(st.number_input("Random seed", value=42, step=1))

                try:
                    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(X)
                except Exception as e:
                    st.error(f"Failed to build preprocessor: {e}")
                    st.exception(traceback.format_exc())
                    preprocessor, numeric_cols, categorical_cols = None, [], []

                st.write(f"Numeric columns: {len(numeric_cols)}; Categorical columns: {len(categorical_cols)}")
                st.write("Dropped cols (heuristic):", dropped_columns)

                quick_train = st.checkbox("Quick train (fewer trees / faster)", value=True)
                if quick_train:
                    rf_params = {"n_estimators": 50, "max_depth": 15, "random_state": random_state, "n_jobs": -1}
                else:
                    rf_params = {"n_estimators": 200, "max_depth": None, "random_state": random_state, "n_jobs": -1}

                if st.button("Run training"):
                    with st.spinner("Training model... this may take a while"):
                        try:
                            strat = y if y.nunique() <= 2 else None
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=random_state, stratify=strat)
                            model = train_model(X_train, y_train, preprocessor, rf_params=rf_params)

                            st.session_state["last_model"] = model
                            st.session_state["last_model_name"] = f"rf_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"

                            results = evaluate_model(model, X_test, y_test)
                            st.success("Training complete")
                            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                            st.metric("Precision", f"{results['precision']*100:.2f}%")
                            st.metric("Recall", f"{results['recall']*100:.2f}%")
                            st.metric("F1-score", f"{results['f1']*100:.2f}%")

                            st.write("Confusion matrix:")
                            st.pyplot(plot_confusion_matrix(results["cm"]))

                            st.write("Classification report:")
                            st.text(classification_report(y_test, results["preds"], zero_division=0))

                            st.write("---")
                            st.subheader("Feature importances (top 30)")
                            fi = get_feature_importances(model, numeric_cols, categorical_cols, top_n=30)
                            if fi is not None:
                                st.bar_chart(fi)
                            else:
                                st.info("Feature importances not available")

                            fn_default = st.session_state.get("last_model_name", "rf_model.pkl")
                            save_path = st.text_input("Filename to save model (e.g., rf_model.pkl)", value=fn_default, key="save_filename")

                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button("Save model to file (server)", key="save_server_btn"):
                                    try:
                                        joblib.dump(model, save_path)
                                        st.success(f"Model saved to server as: {save_path}")
                                        st.info("Note: on some hosts this file may be ephemeral or inaccessible. Use the Download button to get the file locally.")
                                    except Exception as e:
                                        st.error("Failed to save model to server.")
                                        st.exception(traceback.format_exc())

                            with col2:
                                if st.button("Prepare download (in-memory)", key="prepare_download_btn"):
                                    try:
                                        bytes_buf = io.BytesIO()
                                        joblib.dump(model, bytes_buf)
                                        bytes_buf.seek(0)
                                        bdata = bytes_buf.read()
                                        st.download_button(
                                            label="Download model file",
                                            data=bdata,
                                            file_name=save_path,
                                            mime="application/octet-stream",
                                            key="download_model_btn"
                                        )
                                        st.success("Download prepared — click the 'Download model file' button to save locally.")
                                    except Exception:
                                        st.error("Failed to prepare download.")
                                        st.exception(traceback.format_exc())

                        except MemoryError:
                            st.error("MemoryError: dataset too large for this machine. Try sampling a smaller fraction, drop more columns, or use a machine with more RAM.")
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            st.exception(traceback.format_exc())

# ---------- Model Results ----------
if menu == "Model Results":
    st.header("Load and inspect saved model")
    model_file = st.file_uploader("Upload model file (.pkl/.joblib)", type=["pkl", "joblib"], key="model_file")
    test_csv = st.file_uploader("Upload CSV for evaluation (optional)", type=["csv"], key="eval_csv")

    if model_file is not None:
        try:
            model_bytes = model_file.read()
            model = joblib.load(io.BytesIO(model_bytes))
            st.success("Model loaded")
            steps = getattr(model, "steps", None)
            if steps:
                st.write("Pipeline steps:", [s[0] for s in steps])
            expected, numeric_cols_in_model, cat_cols_in_model = infer_model_input_columns(model)
            if expected is None:
                st.warning("Unable to infer model input column names automatically. If your model doesn't store transformer column lists, evaluation may fail. Try training with a pipeline that stores column names.")
                expected = []

            if expected:
                st.info(f"Model expects input columns (first 20 shown): {expected[:20]}")
            else:
                st.info("Model input column list not available.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.exception(traceback.format_exc())
            model = None

        if test_csv is not None and model is not None:
            try:
                try:
                    csv_bytes = test_csv.read()
                    csv_buf = io.StringIO(csv_bytes.decode("utf-8", errors="ignore"))
                    df_test = load_csv_try(csv_buf)
                except Exception:
                    test_csv.seek(0)
                    df_test = load_csv_try(test_csv)
            except Exception as e:
                st.error(f"Failed to read uploaded test CSV: {e}")
                st.exception(traceback.format_exc())
                df_test = None

            if df_test is not None:
                st.success("Test CSV loaded. The app will auto-fill missing expected columns automatically.")
                missing = []
                if expected:
                    missing = [c for c in expected if c not in df_test.columns]
                    if missing:
                        st.warning(f"Missing columns for model: {missing} — auto-filling with defaults.")
                        df_test, added = autofill_missing_columns_for_model(df_test, expected, numeric_cols_in_model)
                        st.info(f"Added columns: {added}")
                else:
                    st.info("No expected column list found; using CSV columns as-is (may fail).")

                st.write("Test CSV loaded. Select target column to evaluate.")
                target_col = st.selectbox("Target column", options=[None] + df_test.columns.tolist(), key="res_target")
                if target_col:
                    sample_vals = df_test[target_col].dropna().unique()[:50].tolist()
                    if len(sample_vals) == 0:
                        st.error("No values found in the selected target column.")
                    else:
                        st.write("Target column sample unique values (first 50):")
                        st.write(sample_vals)

                        try:
                            y_raw = df_test[target_col].copy()
                            X_raw = df_test.drop(columns=[target_col])

                            y_mapped, mapping_used = safe_map_target_using_model(y_raw, model)
                            if mapping_used:
                                st.info(f"Mapped target values using mapping (sample): {list(mapping_used.items())[:6]}")

                            mask = ~pd.isna(y_mapped)
                            if mask.sum() == 0:
                                st.error("No rows remain after mapping the selected target to numeric labels. Choose a different target column or preprocess CSV.")
                                st.stop()

                            X_eval = X_raw.loc[mask].reset_index(drop=True)
                            y_eval = y_mapped.loc[mask].reset_index(drop=True)

                            try:
                                X_prepared, added_cols = prepare_df_for_model(X_eval, model)
                                if added_cols:
                                    st.info(f"Auto-added columns for model: {added_cols}")
                            except Exception as ex:
                                st.warning(f"Prepare-for-model step failed, falling back to simple reindex: {ex}")
                                X_prepared = X_eval.reindex(columns=expected if expected else X_eval.columns, fill_value=0)

                            if X_prepared.shape[0] == 0:
                                st.error("No rows remain for evaluation after preprocessing. Cannot run prediction.")
                                st.stop()

                            for col in X_prepared.columns:
                                if col in numeric_cols_in_model:
                                    X_prepared[col] = pd.to_numeric(X_prepared[col], errors="coerce").fillna(0)
                                else:
                                    if X_prepared[col].dtype == object:
                                        X_prepared[col] = X_prepared[col].apply(lambda x: "" if pd.isna(x) else str(x))

                            try:
                                preds = model.predict(X_prepared)
                            except Exception as pred_err:
                                st.error(f"Model prediction failed: {pred_err}")
                                st.exception(traceback.format_exc())
                                st.stop()

                            if not pd.api.types.is_numeric_dtype(y_eval):
                                try:
                                    y_eval = pd.to_numeric(y_eval, errors="coerce")
                                except Exception:
                                    pass

                            if pd.api.types.is_float_dtype(y_eval) and y_eval.nunique() > 10:
                                st.warning("Selected target appears continuous. Classification metrics skipped.")
                                st.info(f"Predictions made: {len(preds)}")
                                st.write(pd.DataFrame({"pred": preds}).head(20))
                            else:
                                try:
                                    if len(y_eval) != len(preds):
                                        st.warning("Number of predicted rows differs from target rows. Truncating to common length.")
                                        common = min(len(y_eval), len(preds))
                                        y_eval = y_eval.iloc[:common]
                                        preds = preds[:common]

                                    metrics = {}
                                    if len(np.unique(y_eval)) <= 1:
                                        st.warning("Not enough class variety to compute classification metrics.")
                                    else:
                                        metrics["accuracy"] = float(accuracy_score(y_eval, preds))
                                        if len(np.unique(y_eval)) > 2:
                                            metrics["precision"] = float(precision_score(y_eval, preds, average="weighted", zero_division=0))
                                            metrics["recall"] = float(recall_score(y_eval, preds, average="weighted", zero_division=0))
                                            metrics["f1"] = float(f1_score(y_eval, preds, average="weighted", zero_division=0))
                                        else:
                                            metrics["precision"] = float(precision_score(y_eval, preds, zero_division=0))
                                            metrics["recall"] = float(recall_score(y_eval, preds, zero_division=0))
                                            metrics["f1"] = float(f1_score(y_eval, preds, zero_division=0))

                                        st.write("Quick metrics on uploaded test set:")
                                        st.json(metrics)

                                        try:
                                            cm = confusion_matrix(y_eval, preds)
                                            st.pyplot(plot_confusion_matrix(cm))
                                            st.write("Classification report:")
                                            st.text(classification_report(y_eval, preds, zero_division=0))
                                        except Exception:
                                            st.warning("Failed to compute confusion matrix or classification report (maybe shape mismatch).")

                                except Exception as e:
                                    st.error(f"Failed to compute metrics: {e}")
                                    st.exception(traceback.format_exc())

                            out_df = X_raw.reset_index(drop=True).iloc[:len(preds)].copy()
                            out_df["ground_truth"] = y_raw.reset_index(drop=True).iloc[:len(preds)]
                            out_df["prediction"] = preds
                            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            st.exception(traceback.format_exc())

# ---------- Predict ----------
if menu == "Predict":
    st.header("Make predictions")
    st.write("Load a trained model (pickle) and then upload a CSV for batch prediction or provide a single row.")
    model_file = st.file_uploader("Upload trained model (.pkl/.joblib)", type=["pkl", "joblib"], key="predict_model_file")
    if model_file is not None:
        try:
            model = joblib.load(io.BytesIO(model_file.read()))
            st.success("Model loaded")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.exception(traceback.format_exc())
            model = None

        if model is not None:
            st.subheader("Batch prediction (CSV)")
            batch_csv = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_pred")
            if batch_csv is not None:
                try:
                    try:
                        df_batch = load_csv_try(batch_csv)
                    except Exception:
                        batch_csv.seek(0)
                        df_batch = load_csv_try(batch_csv)

                    st.write("Preview:")
                    st.dataframe(df_batch.head())

                    if st.button("Run batch prediction"):
                        expected, numeric_cols_in_model, cat_cols_in_model = infer_model_input_columns(model)
                        if expected:
                            df_batch_prepared, added = prepare_df_for_model(df_batch, model)
                            if added:
                                st.info(f"Auto-added columns for batch: {added}")
                        else:
                            df_batch_prepared = df_batch.copy()
                            for nc in (numeric_cols_in_model or []):
                                if nc in df_batch_prepared.columns:
                                    df_batch_prepared[nc] = pd.to_numeric(df_batch_prepared[nc], errors="coerce").fillna(0)

                        if df_batch_prepared.shape[0] == 0:
                            st.error("No rows to predict after preprocessing.")
                        else:
                            for col in df_batch_prepared.columns:
                                if col in (numeric_cols_in_model or []):
                                    df_batch_prepared[col] = pd.to_numeric(df_batch_prepared[col], errors="coerce").fillna(0)
                                else:
                                    if df_batch_prepared[col].dtype == object:
                                        df_batch_prepared[col] = df_batch_prepared[col].apply(lambda x: "" if pd.isna(x) else str(x))

                            preds = model.predict(df_batch_prepared)
                            df_batch_prepared["predicted_delay"] = preds
                            st.dataframe(df_batch_prepared.head(20))
                            csv_bytes = df_batch_prepared.to_csv(index=False).encode("utf-8")
                            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    st.exception(traceback.format_exc())

            st.subheader("Single-row prediction")
            st.write("Provide comma-separated columns (same order as during training) and values.")
            cols_input = st.text_area("Columns (comma-separated)")
            vals_input = st.text_input("Values (comma-separated)")
            if st.button("Predict single row"):
                if not cols_input or not vals_input:
                    st.error("Provide both columns and values")
                else:
                    cols = [c.strip() for c in cols_input.split(",") if c.strip()]
                    vals = [v.strip() for v in vals_input.split(",")]
                    if len(cols) != len(vals):
                        st.error("Number of values must match number of columns")
                    else:
                        input_df = pd.DataFrame([vals], columns=cols)
                        for c in input_df.columns:
                            try:
                                input_df[c] = pd.to_numeric(input_df[c])
                            except Exception:
                                pass
                        try:
                            expected, numeric_cols_in_model, cat_cols_in_model = infer_model_input_columns(model)
                            if expected:
                                input_df_prepared, added = prepare_df_for_model(input_df, model)
                                if added:
                                    st.info(f"Auto-added columns for single-row input: {added}")
                            else:
                                input_df_prepared = input_df.copy()
                                for nc in (numeric_cols_in_model or []):
                                    if nc in input_df_prepared.columns:
                                        input_df_prepared[nc] = pd.to_numeric(input_df_prepared[nc], errors="coerce").fillna(0)

                            for col in input_df_prepared.columns:
                                if col in (numeric_cols_in_model or []):
                                    input_df_prepared[col] = pd.to_numeric(input_df_prepared[col], errors="coerce").fillna(0)
                                else:
                                    if input_df_prepared[col].dtype == object:
                                        input_df_prepared[col] = input_df_prepared[col].apply(lambda x: "" if pd.isna(x) else str(x))

                            pred = model.predict(input_df_prepared)[0]
                            st.success(f"Predicted label: {pred}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.exception(traceback.format_exc())

# ---------- Export / Load Model ----------
if menu == "Export/Load Model":
    st.header("Save or load model")
    st.write("Training saves a `.pkl` file in the Train page. You can also upload a model file to use in Predict or Model Results pages.")
    st.info("This page is informational — saving is handled in Train page after training completes.")

    model = st.session_state.get("last_model")
    model_name_default = st.session_state.get("last_model_name", "rf_model.pkl")

    if model is None:
        st.warning("No trained model found in current session. Train a model on the Train Model page first, or upload a model on Model Results/Predict pages.")
    else:
        st.success("A recently trained model is available in this session.")
        fn = st.text_input("Filename to save model (e.g., rf_model.pkl)", value=model_name_default, key="export_filename")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Save model to file (server)", key="export_save_server"):
                try:
                    joblib.dump(model, fn)
                    st.success(f"Model saved to server as: {fn}")
                    st.info("Note: on some hosts this file may be ephemeral or inaccessible. Use the Download button to keep a local copy.")
                except Exception as e:
                    st.error("Failed to save model to server.")
                    st.exception(traceback.format_exc())

        with col2:
            if st.button("Download model (.pkl)", key="export_download_prepare"):
                try:
                    bytes_buf = io.BytesIO()
                    joblib.dump(model, bytes_buf)
                    bytes_buf.seek(0)
                    bdata = bytes_buf.read()
                    st.download_button(
                        label="Download model file",
                        data=bdata,
                        file_name=fn,
                        mime="application/octet-stream",
                        key="export_download"
                    )
                    st.success("Download prepared — click the button above to save the file to your computer.")
                except Exception:
                    st.error("Failed to prepare download.")
                    st.exception(traceback.format_exc())

# ---------- About ----------
if menu == "About":
    st.header("About SupplyChainAI")
    st.write("This is an example production-like UI for supply-chain delay prediction.")
    st.write("Company: SupplyChainAI Ltd.")
    st.write("Author: Internal — Original code (no plagiarism).")
    st.markdown("---")
    st.write("How to use:")
    st.write("1. Train & save a sklearn Pipeline that contains a preprocessor (ColumnTransformer) and a classifier.")
    st.write("2. Upload model (.pkl) and a CSV with the same columns (or let the app auto-fill common columns).")
    st.write("3. Download the CSV of predictions from the Model Results or Predict page.")
