# app.py
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# -------- Load artifacts --------
MODEL_PATH = "model.pkl"
FEATURES_PATH = "model_features.json"
DEFAULTS_PATH = "defaults.json"
THRESHOLDS_PATH = "thresholds.json"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # <- NEW

try:
    with open(FEATURES_PATH, "r") as f:
        MODEL_FEATURES = json.load(f)
except FileNotFoundError:
    MODEL_FEATURES = None

try:
    with open(DEFAULTS_PATH, "r") as f:
        DEFAULTS = json.load(f)
except FileNotFoundError:
    DEFAULTS = {}

try:
    with open(THRESHOLDS_PATH, "r") as f:
        THRESHOLDS = json.load(f)
except FileNotFoundError:
    THRESHOLDS = {"t_low": 0.40, "t_high": 0.75}

try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    scaler = None

# -------- Helpers --------
# NOTE: 'married' is NUMERIC (0/1) to match the trained model.
CATEGORICALS = ["gender", "vehicle_type"]
NUMERICS = [
    "age", "vehicle_year", "annual_mileage",
    "credit_score",            # raw input; we build 'credit_score_adj' from this
    "past_accidents", "speeding_violations", "duis", "married"
]

def _build_dataframe_from_form(form):
    row = {}

    # numerics
    for k in NUMERICS:
        v = form.get(k)
        if v is None and k == "duis":
            v = form.get("DUIs")  # tolerate old field name
        if k == "married" and isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ("yes","y","true","1"): v = 1
            elif v_clean in ("no","n","false","0"): v = 0
        try:
            row[k] = float(v)
        except Exception:
            row[k] = np.nan

    # adjusted credit score (model feature)
    cs = row.get("credit_score", np.nan)
    try:
        row["credit_score_adj"] = 1.0 - float(cs)
    except Exception:
        row["credit_score_adj"] = np.nan

    # categoricals
    for k in CATEGORICALS:
        v = form.get(k, "")
        row[k] = (str(v).strip().lower() if v is not None else "")

    df = pd.DataFrame([row])

    # one-hot ONLY categoricals
    X = pd.get_dummies(df, columns=CATEGORICALS, drop_first=False)

    # align to training columns & order
    if MODEL_FEATURES:
        for col in MODEL_FEATURES:
            if col not in X.columns:
                X[col] = 0
        X = X[MODEL_FEATURES]
    return X

def _apply_scaling(X: pd.DataFrame) -> pd.DataFrame:
    if scaler is None:
        return X
    Xs = X.copy()
    cols = list(getattr(scaler, "feature_names_in_", []))
    if cols:
        Xs[cols] = scaler.transform(Xs[cols])
    else:
        num_cols = Xs.select_dtypes(include=[np.number]).columns
        Xs[num_cols] = scaler.transform(Xs[num_cols])
    return Xs

def _risk_bucket(prob):
    if prob < 0.45:
        return "Low Risk", "Your profile indicates a low chance of a claim. Keep up safe driving habits."
    elif prob < 0.70:
        return "Moderate Risk", "There is a moderate chance of a claim. Stay cautious and avoid violations."
    else:
        return "High Risk", "Your profile indicates a high chance of a claim. Please be cautious and avoid violations."

# -------- Routes --------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", defaults=DEFAULTS)

@app.route("/predict", methods=["POST"])
def predict():
    X = _build_dataframe_from_form(request.form)
    X = _apply_scaling(X)  # <- NEW: scale before prediction

    # Probability of CLAIM (class 1)
    probas = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    proba_claim = float(probas[classes.index(1)]) if 1 in classes else float(model.predict_proba(X)[:, 1][0])

    risk_label, msg = _risk_bucket(proba_claim)

    return render_template(
        "result.html",
        prob=proba_claim,
        probability=round(proba_claim * 100, 1),
        risk_label=risk_label,
        message=msg
    )

# --- Debug route (kept; now also scales) ---
@app.route("/_debug_clean")
def _debug_clean():
    form = {
        "age": "30", "gender": "male", "vehicle_year": "2020", "vehicle_type": "sedan",
        "annual_mileage": "8000", "credit_score": "0.9",
        "past_accidents": "0", "speeding_violations": "0", "duis": "0",
        "married": "yes"
    }
    X = _build_dataframe_from_form(form)
    X = _apply_scaling(X)  # <- NEW

    probas = model.predict_proba(X)[0]
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    proba_claim = float(probas[classes.index(1)])

    on_cols = [c for c, v in zip(X.columns.tolist(), X.iloc[0].tolist()) if float(v) == 1.0]
    groups = {}
    for prefix in ["gender_", "vehicle_type_", "married_"]:
        cols = [c for c in X.columns if c.startswith(prefix)]
        groups[prefix] = float(X[cols].iloc[0].sum()) if cols else None

    result = {
        "model_type": str(type(model).__name__),
        "has_calibration": bool(hasattr(model, "calibrated_classifiers_")),
        "classes": classes,
        "prob_claim": round(float(proba_claim), 4),
        "thresholds": {k: float(v) for k, v in THRESHOLDS.items()},
        "on_cols": on_cols[:20],
        "group_sums_should_be_1": groups,
        "has_duis_col": bool("duis" in X.columns)
    }
    result = json.loads(json.dumps(
        result,
        default=lambda x: float(x) if isinstance(x, (np.floating, np.float32, np.float64))
                          else int(x) if isinstance(x, (np.integer, np.int32, np.int64))
                          else x
    ))
    return result

if __name__ == "__main__":
    app.run(debug=True)

