# AI-Powered-Decision-Support-System-ML-Demo

Flask web dashboard demo for basic ML algorithms using the `StudentsPerformance.csv` dataset.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Optional dev tools:

```bash
pip install -r requirements-dev.txt
```

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Streamlit Cloud (recommended deployment)

Set the Streamlit **Main file path** to `streamlit_app.py`, and run:

```bash
streamlit run streamlit_app.py
```

### If you want auto-reload (dev)

```bash
flask --app app run --debug
```

### About `ValueError: signal only works in main thread...`

If you see this error, it usually means the app was started from a non-main thread (for example by a runner like Streamlit or inside a notebook). Running with `python app.py` (as above) or using the Flask CLI command avoids the signal-based reloader issue.

### Login

- Username: `teacher`
- Password: `mldemo`