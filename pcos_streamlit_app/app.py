"""PCOS risk screening prototype (Random Forest + Hybrid feature selection).

Research prototype. It is not a diagnostic tool.
"""

import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "PCOS_Tuned_Random_Forest_Hybrid_FS.joblib"

CONTINUOUS = ["Age", "BMI", "Testosterone_Levelng/dL", "Antral_Follicle_Count"]
HYBRID_FEATURES = [
    "Menstrual_Irregularity",
    "Testosterone_Levelng/dL_Antral_Follicle_Count",
    "Antral_Follicle_Count_Squared",
    "BMI_Antral_Follicle_Count",
    "Antral_Follicle_Count",
    "BMI_Testosterone_Levelng/dL",
    "Testosterone_Levelng/dL",
    "Testosterone_Levelng/dL_Squared",
    "Age_Antral_Follicle_Count",
    "Age",
]

# Physiological limits: values outside are rejected.
HARD_LIMITS = {
    "Age": (10.0, 80.0, "years"),
    "Weight": (20.0, 300.0, "kg"),
    "Height": (100.0, 230.0, "cm"),
    "BMI": (10.0, 70.0, "kg/m²"),
    "Testosterone": (1.0, 500.0, "ng/dL"),
    "Antral follicle count": (0.0, 100.0, ""),
}

# Range of the training data (Step 04 training split): values outside get a warning.
TRAINING_RANGE = {
    "Age": (18.0, 45.0, "years"),
    "BMI": (8.5, 44.7, "kg/m²"),
    "Testosterone": (20.0, 136.4, "ng/dL"),
    "Antral follicle count": (3.0, 39.0, ""),
}

DECISION_THRESHOLD = 0.5
UNCERTAIN_BAND = (0.35, 0.65)

MENSTRUAL_CHOICES = {"Regular": 0, "Irregular": 1}

LANGUAGES = {"English": "en", "বাংলা": "bn"}

TEXT = {
    "en": {
        "title": "PCOS Risk Screening",
        "subtitle": "Research prototype — Random Forest with hybrid feature selection. "
                    "Not a diagnostic tool.",
        "settings": "Settings",
        "language": "Language",
        "basic": "Basic information",
        "clinical": "Clinical information",
        "age": "Age (years)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "menstrual": "Menstrual cycle",
        "select": "Select",
        "Regular": "Regular",
        "Irregular": "Irregular",
        "testosterone": "Testosterone (ng/dL)",
        "afc": "Antral follicle count",
        "predict": "Predict",
        "waiting": "Enter the details and press Predict. The result appears here.",
        "low": "Lower likelihood of PCOS",
        "high": "Higher likelihood of PCOS",
        "estimate": "Estimated probability of PCOS",
        "threshold_note": "The middle mark is the decision threshold (50%).",
        "bmi": "Calculated BMI",
        "tile_model": "Model",
        "tile_time": "Prediction time",
        "tile_features": "Features used",
        "agreement": "Agreement among the {n} trees of the forest: {pct}%",
        "uncertain": "The estimate is close to the decision boundary, so it is uncertain. "
                     "Clinical assessment is needed.",
        "why_title": "About the model",
        "why_body": "Tuned Random Forest on 10 features chosen by hybrid feature selection. "
                    "Training used SMOTE on the training split only. On the held-out test set "
                    "(600 records, used once): accuracy 0.983, F1 0.959, ROC-AUC 0.999.",
        "data_title": "Data",
        "data_body": "4,000 records from two public datasets. The model was frozen before the "
                     "test set was opened.",
        "disclaimer_title": "Disclaimer",
        "disclaimer_body": "Research prototype. It has not been clinically validated and must "
                           "not be used to diagnose or exclude PCOS. If PCOS is suspected, "
                           "consult a qualified clinician.",
        "required": "{field} is required.",
        "zero": "{field} cannot be zero.",
        "finite": "{field} must be a finite number.",
        "range": "{field} must be between {low} and {high} {unit} (entered {value}).",
        "bmi_bad": "The weight and height give a BMI of {bmi} {unit}, which is outside the "
                   "plausible range {low}–{high}. Check weight and height.",
        "train_warn": "{field} ({value}) is outside the range seen in training ({low}–{high}). "
                      "The prediction is less reliable.",
        "fields": {"Age": "Age", "Weight": "Weight", "Height": "Height",
                   "Testosterone": "Testosterone", "Antral follicle count":
                   "Antral follicle count", "Menstrual cycle": "Menstrual cycle",
                   "BMI": "BMI"},
        "units": {"years": "years", "kg": "kg", "cm": "cm", "kg/m²": "kg/m²",
                  "ng/dL": "ng/dL", "": ""},
    },
    "bn": {
        "title": "PCOS ঝুঁকি স্ক্রিনিং",
        "subtitle": "একটি গবেষণা প্রোটোটাইপ — Random Forest ও Hybrid feature selection। "
                    "ক্লিনিক্যাল ডায়াগনসিস টুল নয়।",
        "settings": "সেটিংস",
        "language": "ভাষা",
        "basic": "মৌলিক তথ্য",
        "clinical": "ক্লিনিক্যাল তথ্য",
        "age": "বয়স (বছর)",
        "weight": "ওজন (kg)",
        "height": "উচ্চতা (cm)",
        "menstrual": "মাসিক চক্র",
        "select": "নির্বাচন করুন",
        "Regular": "নিয়মিত",
        "Irregular": "অনিয়মিত",
        "testosterone": "টেস্টোস্টেরন (ng/dL)",
        "afc": "অ্যান্ট্রাল ফলিকল কাউন্ট (AFC)",
        "predict": "পূর্বাভাস দিন (Predict)",
        "waiting": "তথ্য দিয়ে Predict চাপুন — ফলাফল এখানে দেখাবে।",
        "low": "PCOS-এর ঝুঁকি কম বলে অনুমান",
        "high": "PCOS-এর ঝুঁকি বেশি বলে অনুমান",
        "estimate": "PCOS-এর সম্ভাব্যতা",
        "threshold_note": "মাঝের দাগ হলো সিদ্ধান্ত সীমা (৫০%)।",
        "bmi": "গণনা করা BMI",
        "tile_model": "মডেল",
        "tile_time": "Prediction সময়",
        "tile_features": "ব্যবহৃত feature",
        "agreement": "Forest-এর {n}টি tree-র মধ্যে মতৈক্য: {pct}%",
        "uncertain": "ফলাফল সিদ্ধান্ত সীমার কাছাকাছি, তাই অনিশ্চিত। ক্লিনিক্যাল মূল্যায়ন প্রয়োজন।",
        "why_title": "মডেল সম্পর্কে",
        "why_body": "Hybrid feature selection দিয়ে বাছাই করা ১০টি feature-এর উপর Tuned Random "
                    "Forest। প্রশিক্ষণে SMOTE শুধু training অংশে ব্যবহার হয়েছে। আলাদা করে রাখা "
                    "টেস্ট সেটে (৬০০ রেকর্ড, একবারই ব্যবহৃত): accuracy 0.983, F1 0.959, "
                    "ROC-AUC 0.999।",
        "data_title": "ডেটা",
        "data_body": "দুটি পাবলিক ডেটাসেট থেকে ৪,০০০ রেকর্ড। টেস্ট সেট খোলার আগেই মডেল চূড়ান্ত "
                     "করা হয়েছিল।",
        "disclaimer_title": "Disclaimer",
        "disclaimer_body": "গবেষণার জন্য তৈরি প্রোটোটাইপ। ক্লিনিক্যালি যাচাই করা হয়নি এবং PCOS "
                           "নির্ণয় বা বাদ দিতে ব্যবহার করা যাবে না। PCOS সন্দেহ হলে gynecologist-এর "
                           "পরামর্শ নিন।",
        "required": "{field} আবশ্যক।",
        "zero": "{field} শূন্য হতে পারে না।",
        "finite": "{field} একটি বৈধ সংখ্যা হতে হবে।",
        "range": "{field} অবশ্যই {low}–{high} {unit}-এর মধ্যে হতে হবে (দেওয়া হয়েছে {value})।",
        "bmi_bad": "ওজন ও উচ্চতা থেকে BMI {bmi} {unit} হয়েছে, যা সম্ভাব্য সীমা {low}–{high}-এর "
                   "বাইরে। ওজন ও উচ্চতা যাচাই করুন।",
        "train_warn": "{field} ({value}) training ডেটার সীমার ({low}–{high}) বাইরে। পূর্বাভাস কম "
                      "নির্ভরযোগ্য।",
        "fields": {"Age": "বয়স", "Weight": "ওজন", "Height": "উচ্চতা",
                   "Testosterone": "টেস্টোস্টেরন", "Antral follicle count":
                   "অ্যান্ট্রাল ফলিকল কাউন্ট (AFC)", "Menstrual cycle": "মাসিক চক্র", "BMI": "BMI"},
        "units": {"years": "বছর", "kg": "kg", "cm": "cm", "kg/m²": "kg/m²",
                  "ng/dL": "ng/dL", "": ""},
    },
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;600;700&display=swap');
.stApp { font-family: 'Hind Siliguri', 'Source Sans Pro', sans-serif; }
.pcos-hero {
    background: linear-gradient(120deg, #4338ca 0%, #7c3aed 55%, #c026d3 100%);
    color: #ffffff; padding: 22px 26px; border-radius: 16px; margin-bottom: 14px;
}
.pcos-hero h1 { color: #ffffff; margin: 0 0 4px 0; padding: 0; font-size: 1.9rem; }
.pcos-hero p { margin: 0; opacity: 0.92; font-size: 0.98rem; }
.pcos-pill {
    display: inline-block; padding: 3px 14px; border-radius: 999px; margin-bottom: 6px;
    background: linear-gradient(120deg, #4f46e5, #9333ea); color: #ffffff;
    font-weight: 600; font-size: 0.9rem;
}
.pcos-box {
    border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
    background: rgba(127, 127, 127, 0.10); border: 1px solid rgba(127, 127, 127, 0.28);
}
.pcos-box h4 { margin: 0 0 6px 0; font-size: 1rem; }
.pcos-box p { margin: 0; font-size: 0.9rem; opacity: 0.88; line-height: 1.5; }
.pcos-box.accent { border-left: 5px solid #7c3aed; }
.pcos-box.warn { border-left: 5px solid #d97706; }
.pcos-waiting {
    border: 2px dashed rgba(124, 58, 237, 0.55); border-radius: 14px; padding: 26px 16px;
    text-align: center; opacity: 0.85; margin-top: 8px;
}
.pcos-verdict { border-radius: 14px; padding: 18px 20px; font-size: 1.25rem; font-weight: 700; }
.pcos-verdict small { display: block; font-weight: 400; font-size: 0.85rem; opacity: 0.85;
    margin-top: 4px; }
.pcos-verdict.low { background: rgba(5, 150, 105, 0.16); border: 1px solid #059669; }
.pcos-verdict.high { background: rgba(225, 29, 72, 0.14); border: 1px solid #e11d48; }
.pcos-prob { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
.pcos-gauge { position: relative; height: 14px; border-radius: 999px; margin: 14px 0 6px 0;
    background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #e11d48 100%); }
.pcos-marker { position: absolute; top: -5px; width: 6px; height: 24px; border-radius: 3px;
    background: #111827; border: 2px solid #ffffff; transform: translateX(-50%); }
.pcos-tick { position: absolute; top: -3px; width: 2px; height: 20px;
    background: rgba(255, 255, 255, 0.95); transform: translateX(-50%); }
.pcos-tile { border-radius: 12px; padding: 12px 14px; text-align: left;
    background: rgba(124, 58, 237, 0.12); border: 1px solid rgba(124, 58, 237, 0.4); }
.pcos-tile span { display: block; font-size: 0.8rem; opacity: 0.8; }
.pcos-tile b { font-size: 1.35rem; }
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(120deg, #4f46e5, #9333ea); color: #ffffff; border: none;
    font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem;
}
div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.1); color: #ffffff; }
[data-testid="stSidebarHeader"] { height: 0.5rem; min-height: 0; padding: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
[data-testid="stSidebar"] h3 { margin-top: 0; padding-top: 0; }
</style>
"""


# ---------------------------------------------------------------- logic

def compute_bmi(weight_kg, height_cm):
    return float(weight_kg) / (float(height_cm) / 100.0) ** 2


def build_features(age, bmi, testosterone, follicle_count, menstrual_irregularity):
    """Rebuild the engineered feature representation (Step 03) and return
    all 15 features in a dict."""
    base = {
        "Age": float(age),
        "BMI": float(bmi),
        "Testosterone_Levelng/dL": float(testosterone),
        "Antral_Follicle_Count": float(follicle_count),
    }
    features = dict(base)
    for name in CONTINUOUS:
        features[f"{name}_Squared"] = base[name] ** 2
    for first, second in combinations(CONTINUOUS, 2):
        features[f"{first}_{second}"] = base[first] * base[second]
    features["Menstrual_Irregularity"] = int(menstrual_irregularity)
    return features


def model_input(features):
    return pd.DataFrame([[features[name] for name in HYBRID_FEATURES]],
                        columns=HYBRID_FEATURES)


def validate_inputs(age, weight, height, testosterone, follicle_count, menstrual, lang="en"):
    """Return (errors, warnings, bmi). Empty fields and impossible values are
    errors; values outside the training range are warnings."""
    text = TEXT[lang]
    names = text["fields"]
    units = text["units"]
    errors, warnings = [], []

    fields = [("Age", age), ("Weight", weight), ("Height", height),
              ("Testosterone", testosterone),
              ("Antral follicle count", follicle_count)]
    for label, value in fields:
        if value is None:
            errors.append(text["required"].format(field=names[label]))
    if menstrual not in MENSTRUAL_CHOICES:
        errors.append(text["required"].format(field=names["Menstrual cycle"]))
    if errors:
        return errors, warnings, None

    for label, value in [("Age", age), ("Weight", weight), ("Height", height)]:
        if value == 0:
            errors.append(text["zero"].format(field=names[label]))
    for label, value in fields:
        if not np.isfinite(value):
            errors.append(text["finite"].format(field=names[label]))
    if errors:
        return errors, warnings, None

    for label, value in fields:
        low, high, unit = HARD_LIMITS[label]
        if not low <= value <= high:
            errors.append(text["range"].format(
                field=names[label], low=f"{low:g}", high=f"{high:g}",
                unit=units[unit], value=f"{value:g}").replace(" )", ")").replace("  ", " "))
    if errors:
        return errors, warnings, None

    bmi = compute_bmi(weight, height)
    low, high, unit = HARD_LIMITS["BMI"]
    if not low <= bmi <= high:
        errors.append(text["bmi_bad"].format(bmi=f"{bmi:.1f}", unit=units[unit],
                                             low=f"{low:g}", high=f"{high:g}"))
        return errors, warnings, bmi

    checks = [("Age", age), ("BMI", bmi), ("Testosterone", testosterone),
              ("Antral follicle count", follicle_count)]
    for label, value in checks:
        low, high, unit = TRAINING_RANGE[label]
        if not low <= value <= high:
            warnings.append(text["train_warn"].format(
                field=names[label], value=f"{value:g}", low=f"{low:g}", high=f"{high:g}"))
    return errors, warnings, bmi


def predict(model, age, bmi, testosterone, follicle_count, menstrual):
    features = build_features(age, bmi, testosterone, follicle_count,
                              MENSTRUAL_CHOICES[menstrual])
    frame = model_input(features)
    probability = float(model.predict_proba(frame)[0, 1])
    tree_votes = np.array([tree.predict(frame.to_numpy())[0] for tree in model.estimators_])
    tree_agreement = float(max(tree_votes.mean(), 1 - tree_votes.mean()))
    return probability, tree_agreement


def describe(probability):
    positive = probability >= DECISION_THRESHOLD
    uncertain = UNCERTAIN_BAND[0] <= probability <= UNCERTAIN_BAND[1]
    return positive, uncertain


# ------------------------------------------------------------------- UI

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    if hasattr(model, "feature_names_in_"):
        assert list(model.feature_names_in_) == HYBRID_FEATURES, "Model feature order mismatch."
    return model


def pill(label):
    st.markdown(f'<span class="pcos-pill">{label}</span>', unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        current = st.session_state.get("language", "English")
        text = TEXT[LANGUAGES.get(current, "en")]
        st.markdown(f"### ⚙️ {text['settings']}")
        choice = st.radio(text["language"], list(LANGUAGES), horizontal=True,
                          key="language")
        st.markdown(
            f'<div class="pcos-box accent"><h4>ℹ️ {text["why_title"]}</h4>'
            f'<p>{text["why_body"]}</p></div>'
            f'<div class="pcos-box accent"><h4>📦 {text["data_title"]}</h4>'
            f'<p>{text["data_body"]}</p></div>'
            f'<div class="pcos-box warn"><h4>⚠️ {text["disclaimer_title"]}</h4>'
            f'<p>{text["disclaimer_body"]}</p></div>',
            unsafe_allow_html=True)
    return LANGUAGES[choice]


def show_result(model, text, age, bmi, testosterone, follicle_count, menstrual, warnings):
    started = time.perf_counter()
    probability, agreement = predict(model, age, bmi, testosterone, follicle_count, menstrual)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    positive, uncertain = describe(probability)

    st.info(f"{text['bmi']}: {bmi:.1f} kg/m²")
    for message in warnings:
        st.warning(message)

    percent = probability * 100.0
    verdict_class = "high" if positive else "low"
    icon = "⚠️" if positive else "✅"
    verdict_text = text["high"] if positive else text["low"]

    left, right = st.columns([2, 3])
    left.markdown(
        f'<div class="pcos-verdict {verdict_class}">{icon} {verdict_text}'
        f'<small>{text["agreement"].format(n=len(model.estimators_), pct=f"{agreement * 100:.0f}")}'
        f'</small></div>', unsafe_allow_html=True)
    right.markdown(
        f'<div class="pcos-box"><span>{text["estimate"]}</span>'
        f'<div class="pcos-prob">{percent:.1f}%</div>'
        f'<div class="pcos-gauge"><div class="pcos-tick" style="left:'
        f'{DECISION_THRESHOLD * 100:.0f}%"></div>'
        f'<div class="pcos-marker" style="left:{min(max(percent, 1.0), 99.0):.1f}%"></div></div>'
        f'<p>{text["threshold_note"]}</p></div>', unsafe_allow_html=True)

    tiles = st.columns(3)
    tile_values = [(text["tile_model"], "Random Forest"),
                   (text["tile_time"], f"{elapsed_ms:.1f} ms"),
                   (text["tile_features"], str(len(HYBRID_FEATURES)))]
    for column, (label, value) in zip(tiles, tile_values):
        column.markdown(f'<div class="pcos-tile"><span>{label}</span><b>{value}</b></div>',
                        unsafe_allow_html=True)

    if uncertain:
        st.warning(text["uncertain"])


def main():
    st.set_page_config(page_title="PCOS Risk Screening", page_icon="🩺", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    lang = sidebar()
    text = TEXT[lang]
    model = load_model()

    st.markdown(
        f'<div class="pcos-hero"><h1>🩺 {text["title"]}</h1><p>{text["subtitle"]}</p></div>',
        unsafe_allow_html=True)

    with st.form("patient_form"):
        with st.container(border=True):
            pill(text["basic"])
            col_age, col_weight, col_height = st.columns(3)
            age = col_age.number_input(text["age"], value=None, step=1, format="%d",
                                       placeholder="28", key="age")
            weight = col_weight.number_input(text["weight"], value=None, step=0.1,
                                             format="%.1f", placeholder="62.0", key="weight")
            height = col_height.number_input(text["height"], value=None, step=0.1,
                                             format="%.1f", placeholder="160.0", key="height")
        with st.container(border=True):
            pill(text["clinical"])
            col_cycle, col_testosterone, col_follicles = st.columns(3)
            menstrual = col_cycle.selectbox(
                text["menstrual"], list(MENSTRUAL_CHOICES), index=None,
                format_func=lambda key: text[key], placeholder=text["select"], key="menstrual")
            testosterone = col_testosterone.number_input(
                text["testosterone"], value=None, step=0.1, format="%.1f",
                placeholder="55.0", key="testosterone")
            follicle_count = col_follicles.number_input(
                text["afc"], value=None, step=1, format="%d", placeholder="14",
                key="follicles")
        submitted = st.form_submit_button(text["predict"], use_container_width=True)

    entered = dict(age=age, weight=weight, height=height, testosterone=testosterone,
                   follicle_count=follicle_count, menstrual=menstrual)
    if submitted:
        st.session_state["last_inputs"] = entered
    payload = st.session_state.get("last_inputs")

    if payload is None:
        st.markdown(f'<div class="pcos-waiting">{text["waiting"]}</div>',
                    unsafe_allow_html=True)
    else:
        errors, warnings, bmi = validate_inputs(
            payload["age"], payload["weight"], payload["height"], payload["testosterone"],
            payload["follicle_count"], payload["menstrual"], lang)
        if errors:
            st.session_state.pop("last_inputs", None)
            for message in errors:
                st.error(message)
        else:
            show_result(model, text, payload["age"], bmi, payload["testosterone"],
                        payload["follicle_count"], payload["menstrual"], warnings)

    st.markdown(
        f'<div class="pcos-box warn" style="margin-top:16px"><p>⚠️ '
        f'{text["disclaimer_body"]}</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
