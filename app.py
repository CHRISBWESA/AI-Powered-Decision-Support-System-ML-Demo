from flask import Flask, render_template, request, redirect, url_for # pyright: ignore[reportMissingImports]
import os
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
import pandas as pd # pyright: ignore[reportMissingModuleSource]

from prepare_data import (
    X_train_class, X_test_class, y_train_class, y_test_class,
    X_train_reg, X_test_reg, y_train_reg, y_test_reg,
    X, y_class, y_reg
)

from sklearn.linear_model import LinearRegression, LogisticRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeClassifier # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor # pyright: ignore[reportMissingModuleSource]
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import KNeighborsClassifier # pyright: ignore[reportMissingModuleSource]
from sklearn.naive_bayes import GaussianNB # pyright: ignore[reportMissingModuleSource]
from sklearn.cluster import KMeans, DBSCAN # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # pyright: ignore[reportMissingModuleSource]

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOGIN ----------------
@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'teacher' and password == 'mldemo':
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

# ---------------- HOME ----------------
@app.route('/home')
def home():
    return render_template('home.html')

# ---------------- UPLOAD ----------------
@app.route('/upload', methods=['GET','POST'])
def upload():
    message = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            message = f"File '{file.filename}' uploaded successfully!"
        else:
            message = "Invalid file. Please upload a CSV."
    return render_template('upload.html', message=message)

# ---------------- ALGORITHMS ----------------
@app.route('/algorithm/<name>', methods=['GET','POST'])
def algorithm(name):
    result = None
    name_lower = name.lower()

    # ---------------- REGRESSION ----------------
    if name_lower == 'linear':
        model = LinearRegression()
        model.fit(X_train_reg, y_train_reg)
        score = round(model.score(X_test_reg, y_test_reg), 2)
        y_pred = model.predict(X_test_reg)
        # Plot actual vs predicted
        plt.figure(figsize=(6,4))
        plt.scatter(y_test_reg, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linear Regression: Actual vs Predicted")
        plot_path = "linear_plot.png"
        plt.savefig(f"static/{plot_path}")
        plt.close()
        result = {"score": score, "plot": plot_path}

    elif name_lower == 'gradient_boosting':
        model = GradientBoostingRegressor()
        model.fit(X_train_reg, y_train_reg)
        score = round(model.score(X_test_reg, y_test_reg), 2)
        y_pred = model.predict(X_test_reg)
        plt.figure(figsize=(6,4))
        plt.scatter(y_test_reg, y_pred, color='green', alpha=0.5)
        plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Gradient Boosting: Actual vs Predicted")
        plot_path = "gradient_plot.png"
        plt.savefig(f"static/{plot_path}")
        plt.close()
        result = {"score": score, "plot": plot_path}

    # ---------------- CLASSIFICATION ----------------
    elif name_lower == 'logistic':
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        # Confusion Matrix
        y_pred = model.predict(X_test_class)
        cm = confusion_matrix(y_test_class, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues')
        plot_path = "logistic_plot.png"
        plt.savefig(f"static/{plot_path}")
        plt.close()
        result = {"score": score, "plot": plot_path}

    elif name_lower == 'decision_tree':
        model = DecisionTreeClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        result = score

    elif name_lower == 'random_forest':
        model = RandomForestClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        result = score

    elif name_lower == 'svm':
        model = SVC()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        result = score

    elif name_lower == 'knn':
        model = KNeighborsClassifier()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        result = score

    elif name_lower == 'naive_bayes':
        model = GaussianNB()
        model.fit(X_train_class, y_train_class)
        score = round(model.score(X_test_class, y_test_class), 2)
        result = score

    # ---------------- CLUSTERING ----------------
    elif name_lower == 'kmeans':
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        result = f'Labels: {model.labels_[:10]} ...'
        # Optional: plot clusters
        plt.figure(figsize=(6,4))
        plt.scatter(X.iloc[:,0], X.iloc[:,1], c=model.labels_, cmap='viridis', alpha=0.6)
        plt.title("KMeans Clustering")
        plot_path = "kmeans_plot.png"
        plt.savefig(f"static/{plot_path}")
        plt.close()
        result = {"result": result, "plot": plot_path}

    elif name_lower == 'dbscan':
        model = DBSCAN()
        model.fit(X)
        result = f'Labels: {model.labels_[:10]} ...'
        # Optional: plot clusters
        plt.figure(figsize=(6,4))
        plt.scatter(X.iloc[:,0], X.iloc[:,1], c=model.labels_, cmap='plasma', alpha=0.6)
        plt.title("DBSCAN Clustering")
        plot_path = "dbscan_plot.png"
        plt.savefig(f"static/{plot_path}")
        plt.close()
        result = {"result": result, "plot": plot_path}

    return render_template(f'algorithms/{name_lower}.html', result=result)

if __name__ == '__main__':
    # NOTE:
    # - Werkzeug's reloader uses OS signals, which crash if the app is started
    #   from a non-main thread (e.g. when executed inside Streamlit/Jupyter).
    # - For local development with auto-reload, prefer:
    #     flask --app app run --debug
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
