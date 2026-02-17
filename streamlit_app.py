from __future__ import annotations

import hashlib
import io

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import streamlit as st  # pyright: ignore[reportMissingImports]
from sklearn.cluster import DBSCAN, KMeans  # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier  # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression, LogisticRegression  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # pyright: ignore[reportMissingModuleSource]
from sklearn.naive_bayes import GaussianNB  # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import KNeighborsClassifier  # pyright: ignore[reportMissingModuleSource]
from sklearn.svm import SVC  # pyright: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeClassifier  # pyright: ignore[reportMissingModuleSource]

from data_pipeline import DatasetError, PreparedData, prepare_datasets


st.set_page_config(page_title="ML Dashboard", layout="wide")


USERNAME = "teacher"
PASSWORD = "mldemo"


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _prepare_from_bytes(data_bytes: bytes) -> PreparedData:
    # BytesIO lets us reuse the same pipeline without writing to disk.
    return prepare_datasets(io.BytesIO(data_bytes))


@st.cache_data(show_spinner=False)
def _run_algorithm(dataset_bytes: bytes, algo_key: str) -> dict:
    data = _prepare_from_bytes(dataset_bytes)

    X_train_class, X_test_class = data.X_train_class, data.X_test_class
    y_train_class, y_test_class = data.y_train_class, data.y_test_class
    X_train_reg, X_test_reg = data.X_train_reg, data.X_test_reg
    y_train_reg, y_test_reg = data.y_train_reg, data.y_test_reg
    X = data.X

    # ---------------- REGRESSION ----------------
    if algo_key == "linear":
        model = LinearRegression()
        model.fit(X_train_reg, y_train_reg)
        score = round(model.score(X_test_reg, y_test_reg), 2)
        y_pred = model.predict(X_test_reg)

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(y_test_reg, y_pred, color="blue", alpha=0.5)
        plt.plot(
            [y_test_reg.min(), y_test_reg.max()],
            [y_test_reg.min(), y_test_reg.max()],
            "r--",
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linear Regression: Actual vs Predicted")
        return {"score_label": "R²", "score": score, "plot_png": _fig_to_png_bytes(fig)}

    if algo_key == "gradient_boosting":
        model = GradientBoostingRegressor()
        model.fit(X_train_reg, y_train_reg)
        score = round(model.score(X_test_reg, y_test_reg), 2)
        y_pred = model.predict(X_test_reg)

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(y_test_reg, y_pred, color="green", alpha=0.5)
        plt.plot(
            [y_test_reg.min(), y_test_reg.max()],
            [y_test_reg.min(), y_test_reg.max()],
            "r--",
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Gradient Boosting: Actual vs Predicted")
        return {"score_label": "R²", "score": score, "plot_png": _fig_to_png_bytes(fig)}

    # ---------------- CLASSIFICATION ----------------
    if algo_key == "logistic":
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)

        y_pred = model.predict(X_test_class)
        cm = confusion_matrix(y_test_class, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title("Logistic Regression: Confusion Matrix")
        return {"score_label": "Accuracy", "score": score, "plot_png": _fig_to_png_bytes(fig)}

    if algo_key == "decision_tree":
        model = DecisionTreeClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        return {"score_label": "Accuracy", "score": score}

    if algo_key == "random_forest":
        model = RandomForestClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        return {"score_label": "Accuracy", "score": score}

    if algo_key == "svm":
        model = SVC()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        return {"score_label": "Accuracy", "score": score}

    if algo_key == "knn":
        model = KNeighborsClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        return {"score_label": "Accuracy", "score": score}

    if algo_key == "naive_bayes":
        model = GaussianNB()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        return {"score_label": "Accuracy", "score": score}

    # ---------------- CLUSTERING ----------------
    if algo_key == "kmeans":
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        labels_preview = f"Labels preview: {model.labels_[:10].tolist()} ..."

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=model.labels_, cmap="viridis", alpha=0.6)
        plt.title("KMeans Clustering (first 2 features)")
        return {"text": labels_preview, "plot_png": _fig_to_png_bytes(fig)}

    if algo_key == "dbscan":
        model = DBSCAN()
        model.fit(X)
        labels_preview = f"Labels preview: {model.labels_[:10].tolist()} ..."

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=model.labels_, cmap="plasma", alpha=0.6)
        plt.title("DBSCAN Clustering (first 2 features)")
        return {"text": labels_preview, "plot_png": _fig_to_png_bytes(fig)}

    raise ValueError(f"Unknown algorithm: {algo_key}")


def _login_ui() -> None:
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == USERNAME and password == PASSWORD:
            st.session_state["authed"] = True
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Invalid credentials.")


def _main_ui() -> None:
    st.title("ML Dashboard")

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_right:
        st.subheader("Session")
        if st.button("Logout"):
            for k in ["authed", "dataset_bytes", "dataset_name", "dataset_hash"]:
                st.session_state.pop(k, None)
            st.rerun()

        dataset_name = st.session_state.get("dataset_name")
        if dataset_name:
            st.caption(f"Active dataset: {dataset_name}")
        else:
            st.warning("No dataset uploaded yet.")

    with col_left:
        st.subheader("1) Upload CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            data_bytes = uploaded.getvalue()
            dataset_hash = hashlib.md5(data_bytes).hexdigest()
            try:
                _prepare_from_bytes(data_bytes)
            except DatasetError as e:
                st.error(str(e))
            except Exception as e:  # pragma: no cover
                st.error(f"Could not read CSV: {e}")
            else:
                st.session_state["dataset_bytes"] = data_bytes
                st.session_state["dataset_name"] = uploaded.name
                st.session_state["dataset_hash"] = dataset_hash
                st.success("Dataset uploaded and validated.")

        st.divider()

        st.subheader("2) Run an algorithm")
        algorithms = {
            "Linear Regression": "linear",
            "Gradient Boosting (Regressor)": "gradient_boosting",
            "Logistic Regression": "logistic",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
            "SVM": "svm",
            "KNN": "knn",
            "Naive Bayes": "naive_bayes",
            "KMeans": "kmeans",
            "DBSCAN": "dbscan",
        }

        algo_label = st.selectbox("Select algorithm", list(algorithms.keys()))
        run_disabled = "dataset_bytes" not in st.session_state
        if st.button("Run", disabled=run_disabled):
            dataset_bytes = st.session_state["dataset_bytes"]
            algo_key = algorithms[algo_label]
            with st.spinner("Training / running algorithm..."):
                out = _run_algorithm(dataset_bytes, algo_key)

            st.subheader("Result")
            if "score" in out:
                st.metric(out.get("score_label", "Score"), out["score"])
            if "text" in out:
                st.write(out["text"])
            if "plot_png" in out:
                st.image(out["plot_png"], caption=algo_label, use_container_width=False)
        elif run_disabled:
            st.info("Upload a CSV first to enable algorithms.")


def main() -> None:
    if "authed" not in st.session_state:
        st.session_state["authed"] = False

    if not st.session_state["authed"]:
        st.title("ML Dashboard")
        _login_ui()
        return

    _main_ui()


if __name__ == "__main__":
    main()

