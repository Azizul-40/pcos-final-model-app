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

# Accepted input limits (wider than the dataset so more people can use the app).
# Values outside the dataset range below are accepted but come with a reliability warning.
HARD_LIMITS = {
    "Age": (12.0, 60.0, "years"),
    "Weight": (25.0, 200.0, "kg"),
    "Height": (120.0, 210.0, "cm"),
    "BMI": (10.0, 60.0, "kg/m²"),
    "Testosterone": (5.0, 200.0, "ng/dL"),
    "Antral follicle count": (0.0, 60.0, ""),
}

# Range of the 4,000-record dataset: values outside get a warning.
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
        "wait_title": "Ready when you are",
        "guide_title": "Risk guide",
        "band1": "Low likelihood of PCOS", "band2": "Medium likelihood of PCOS",
        "band3": "High likelihood of PCOS",
        "lean_pos": "The model leans towards PCOS.",
        "lean_neg": "The model leans towards no PCOS.",
        "guide_note": "In the medium range the model is unsure; a clinical assessment is needed.",
        "step1": "Enter details", "step2": "Press Predict", "step3": "See the result",
        "language": "Language",
        "theme": "Theme", "theme_System": "💻 System", "theme_Light": "☀️ Light", "theme_Dark": "🌙 Dark",
        "text_size": "Text size", "size_Normal": "Normal", "size_Large": "Large",
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
        "train_item": "{field} {value} (training range {low}–{high})",
        "train_all": "Outside the range seen in training: {items}. The prediction is less reliable.",
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
        "wait_title": "প্রস্তুত",
        "guide_title": "ঝুঁকির নির্দেশিকা",
        "band1": "PCOS-এর ঝুঁকি কম", "band2": "PCOS-এর ঝুঁকি মাঝারি",
        "band3": "PCOS-এর ঝুঁকি বেশি",
        "lean_pos": "মডেল PCOS-এর দিকে ঝুঁকছে।",
        "lean_neg": "মডেল PCOS না হওয়ার দিকে ঝুঁকছে।",
        "guide_note": "মাঝারি ধাপে মডেল নিজেই নিশ্চিত নয়; ক্লিনিক্যাল মূল্যায়ন প্রয়োজন।",
        "step1": "তথ্য দিন", "step2": "Predict চাপুন", "step3": "ফলাফল দেখুন",
        "language": "ভাষা",
        "theme": "থিম", "theme_System": "💻 সিস্টেম", "theme_Light": "☀️ সাদা", "theme_Dark": "🌙 কালো",
        "text_size": "লেখার আকার", "size_Normal": "সাধারণ", "size_Large": "বড়",
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
        "train_item": "{field} {value} (training সীমা {low}–{high})",
        "train_all": "Training ডেটার সীমার বাইরে: {items}। পূর্বাভাস কম নির্ভরযোগ্য।",
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
    color: #ffffff; padding: 16px 24px; border-radius: 14px; margin-bottom: 6px;
}
.pcos-hero h1 { color: #ffffff; margin: 0 0 2px 0; padding: 0; font-size: 1.6rem; }
.pcos-hero p { margin: 0; opacity: 1; font-size: 0.95rem; }
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
.pcos-box p { margin: 0; font-size: 0.92rem; opacity: 1; line-height: 1.55; }
.pcos-box.accent { border-left: 5px solid #7c3aed; }
.pcos-box.warn { border-left: 5px solid #d97706; }
.pcos-waiting { border: 1px solid var(--border); background: var(--field); border-radius: 14px;
    padding: 20px 16px; text-align: center; margin-top: 4px; }
.pcos-waiting .ico { width: 54px; height: 54px; border-radius: 50%; margin: 0 auto 8px auto;
    display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
    background: linear-gradient(135deg, #4f46e5, #7c3aed); }
.pcos-waiting h3 { margin: 0 0 2px 0; padding: 0; font-size: 1.25rem; }
.pcos-waiting p { margin: 0 0 12px 0; color: var(--muted); }
.pcos-steps { display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap; }
.pcos-step { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
    border-radius: 999px; border: 1px solid var(--border); background: var(--card); font-weight: 600;
    font-size: 0.92rem; }
.pcos-step i { font-style: normal; width: 22px; height: 22px; border-radius: 50%; color: #ffffff;
    background: #4f46e5; display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.8rem; }
.pcos-arrow { opacity: 0.6; }
.pcos-verdict { border-radius: 14px; padding: 18px 20px; font-size: 1.25rem; font-weight: 700; }
.pcos-verdict small { display: block; font-weight: 400; font-size: 0.85rem; opacity: 0.85;
    margin-top: 4px; }
.pcos-verdict.low { background: rgba(5, 150, 105, 0.16); border: 1px solid #059669; }
.pcos-verdict.high { background: rgba(225, 29, 72, 0.14); border: 1px solid #e11d48; }
.pcos-guide-row { display: flex; align-items: flex-start; gap: 10px; margin: 8px 0; }
.pcos-guide-row > div { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.pcos-guide-row b { font-size: 0.9rem; }
.pcos-guide-text { font-size: 0.82rem; line-height: 1.35; }
.pcos-swatch { width: 34px; height: 16px; border-radius: 5px; flex: none; margin-top: 3px; }
.pcos-prob { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
.pcos-result { display: grid; grid-template-columns: 2fr 3fr; gap: 12px; align-items: stretch;
    margin: 4px 0 12px 0; }
.pcos-result > div { margin: 0; height: 100%; box-sizing: border-box; }
.pcos-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 0 0 12px 0; }
.pcos-tiles .pcos-tile { box-sizing: border-box; min-height: 74px; }
.pcos-verdict, .pcos-verdict * { color: var(--fg); }
@media (max-width: 760px) { .pcos-result, .pcos-tiles { grid-template-columns: 1fr; } }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"], .block-container { padding: 3rem 0.8rem 1rem 0.8rem; }
    .pcos-hero { padding: 12px 14px; }
    .pcos-hero h1 { font-size: 1.3rem; }
    .pcos-hero p { font-size: 0.85rem; }
    .pcos-verdict { font-size: 1.05rem; padding: 14px 16px; }
    .pcos-prob { font-size: 1.8rem; }
}
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
[data-testid="stSidebarHeader"] { height: 2.2rem; min-height: 0; padding: 0.3rem 0.8rem; }
[data-testid="stSidebarUserContent"] { padding-top: 0; }
[data-testid="stSidebar"] *, [data-testid="stSidebarContent"] { scrollbar-width: none; -ms-overflow-style: none; }
[data-testid="stSidebar"] *::-webkit-scrollbar, [data-testid="stSidebarContent"]::-webkit-scrollbar { display: none; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; padding: 0.4rem 0.2rem 0.2rem 0.2rem; }
.pcos-hero, .pcos-waiting, .pcos-box { width: 100%; box-sizing: border-box; }
[data-testid="stMainBlockContainer"], .block-container { padding: 3.2rem 2.2rem 1rem 2.2rem; max-width: 1200px; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
[data-testid="stSidebar"] .pcos-box { background: var(--card); border: 1px solid var(--border);
    border-left-width: 4px; border-radius: 12px; padding: 10px 14px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.10); }
[data-testid="stSidebar"] .pcos-box p { font-size: 0.84rem; line-height: 1.45; }
[data-testid="stSidebar"] .pcos-box h4 { font-size: 0.98rem; margin: 0 0 6px 0; padding-bottom: 6px;
    border-bottom: 1px solid var(--border); }

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
            warnings.append(text["train_item"].format(
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
    uncertain = 35 <= round(probability * 100) <= 65
    return positive, uncertain


# ------------------------------------------------------------------- UI

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    if hasattr(model, "feature_names_in_"):
        assert list(model.feature_names_in_) == HYBRID_FEATURES, "Model feature order mismatch."
    return model


LIGHT_VARS = ("#ffffff", "#f1f5f9", "#000000", "#f5f3ff", "#c4b5fd", "#ffffff", "#6b6b8a", "#ede9fe", "brightness(0)")
DARK_VARS = ("#0b0f19", "#111827", "#ffffff", "#1f2937", "#4b5563", "#1b2433", "#9ca3af", "#374151", "brightness(0) invert(1)")


def _vars(v):
    names = ("bg", "side", "fg", "field", "border", "card", "muted", "btn", "icon")
    return "".join(f"--{n}:{x};" for n, x in zip(names, v))


def apply_display_settings():
    theme = prefs()["theme"]
    size = prefs()["text_size"]
    scale = "115%" if size == "Large" else "100%"
    if theme == "Dark":
        root = f":root{{{_vars(DARK_VARS)}}}"
    elif theme == "Light":
        root = f":root{{{_vars(LIGHT_VARS)}}}"
    else:
        root = (f":root{{{_vars(LIGHT_VARS)}}}"
                f"@media (prefers-color-scheme: dark){{:root{{{_vars(DARK_VARS)}}}}}")
    css = root + f"""
html {{ font-size: {scale}; }}
[data-testid="stMain"], section.main, .stApp {{ scrollbar-width: none; -ms-overflow-style: none; }}
[data-testid="stMain"]::-webkit-scrollbar, section.main::-webkit-scrollbar,
.stApp::-webkit-scrollbar {{ display: none; }}
.stApp, [data-testid="stHeader"] {{ background: var(--bg); }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: var(--side); }}
.stApp :is(p, label, span, li, h1, h2, h3, h4, h5, small, div[data-testid="stMarkdownContainer"]):not(.pcos-hero *):not(.pcos-verdict *):not(.pcos-pill):not(button *) {{ color: var(--fg); }}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] > div,
[data-baseweb="base-input"] > div, [data-testid="stNumberInputContainer"],
.stApp [data-baseweb="select"] > div {{
    background: var(--field) !important; color: var(--fg) !important; }}
[data-baseweb="input"], [data-testid="stNumberInputContainer"], .stApp [data-baseweb="select"] > div {{
    border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden; }}
[data-baseweb="input"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within {{
    border-color: #4f46e5 !important; box-shadow: 0 0 0 2px rgba(79,70,229,0.25); }}
.stApp input {{ background: var(--field) !important; color: var(--fg) !important;
    -webkit-text-fill-color: var(--fg) !important; caret-color: var(--fg); }}
.stApp input::placeholder {{ color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1 !important; }}
.stApp [data-baseweb="select"] * {{ color: var(--fg) !important; }}
.stApp [data-baseweb="select"] div {{ background: var(--field) !important; }}
.stApp [data-baseweb="select"] svg {{ fill: var(--fg) !important; }}
.stApp [data-testid="stNumberInput"] button {{ background: var(--btn) !important; color: var(--fg) !important; border: none !important; }}
.stApp [data-testid="stNumberInput"] button svg {{ fill: var(--fg) !important; }}
[data-testid="stToolbar"], [data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {{ filter: var(--icon); }}
[data-testid="stToolbar"] *, [data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"] * {{ color: #000000 !important; opacity: 1 !important; }}
[data-testid="stPopover"] button {{
    background: var(--field) !important; color: var(--fg) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    min-height: 3rem; padding: 0.5rem 1rem !important; }}
[data-testid="stPopover"] button, [data-testid="stPopover"] button * {{
    font-size: 1.2rem !important; font-weight: 700 !important; color: var(--fg) !important; }}
[data-testid="stPopover"] button svg {{ width: 1.3rem; height: 1.3rem; }}
[data-baseweb="popover"] > div, [data-baseweb="popover"] li, [data-baseweb="menu"] {{
    background: var(--card); color: var(--fg); }}
[data-testid="stPopoverBody"] {{ background: var(--card) !important;
    border: 1px solid var(--border) !important; border-radius: 12px !important; }}
[data-testid="stPopoverBody"], [data-testid="stPopoverBody"] * {{ color: var(--fg) !important; }}
[data-testid="stPopoverBody"] [data-baseweb="radio"]:has(input:not(:checked)) > div:first-child {{
    background: transparent !important; border-color: var(--muted) !important; }}
[data-baseweb="popover"] :is(p, label, span, div) {{ color: var(--fg); }}"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def pill(label):
    st.markdown(f'<span class="pcos-pill">{label}</span>', unsafe_allow_html=True)


BANDS = ((0, 34, 17), (35, 65, 50), (66, 100, 83))


def risk_band(probability):
    """1 = low (0-34%), 2 = medium (35-65%), 3 = high (66-100%)."""
    percent = round(probability * 100)
    if percent < 35:
        return 1
    if percent <= 65:
        return 2
    return 3


def risk_guide_html(text):
    rows = ""
    for number, (low, high, middle) in enumerate(BANDS, start=1):
        background, border = risk_style(middle)
        rows += (f'<div class="pcos-guide-row"><span class="pcos-swatch" '
                 f'style="background:{background};border:1px solid {border}"></span>'
                 f'<div><b>{low}–{high}%</b>'
                 f'<div class="pcos-guide-text">{text["band" + str(number)]}</div></div></div>')
    return (f'<div class="pcos-box accent"><h4>🎯 {text["guide_title"]}</h4>{rows}'
            f'<p>{text["guide_note"]}</p></div>')


def sidebar(text, lang):
    with st.sidebar:
        settings_popover(text, lang)
        st.markdown(
            risk_guide_html(text) +
            f'<div class="pcos-box accent"><h4>ℹ️ {text["why_title"]}</h4>'
            f'<p>{text["why_body"]}</p></div>'
            f'<div class="pcos-box accent"><h4>📦 {text["data_title"]}</h4>'
            f'<p>{text["data_body"]}</p></div>'
            f'<div class="pcos-box warn"><h4>⚠️ {text["disclaimer_title"]}</h4>'
            f'<p>{text["disclaimer_body"]}</p></div>',
            unsafe_allow_html=True)


PREF_DEFAULTS = {"language": "English", "theme": "System", "text_size": "Normal"}


def prefs():
    """The user's display choices, kept outside the widgets so a language change
    can never reset them."""
    return st.session_state.setdefault("_prefs", dict(PREF_DEFAULTS))


def _saver(pref, widget_key):
    def save():
        prefs()[pref] = st.session_state[widget_key]
    return save


def settings_popover(text, lang):
    saved = prefs()
    languages = list(LANGUAGES)
    themes = ["System", "Light", "Dark"]
    sizes = ["Normal", "Large"]
    with st.popover("⚙️ Settings", use_container_width=True):
        st.markdown(f"**{text['language']}**")
        st.radio("language_choice", languages, index=languages.index(saved["language"]),
                 key="language_widget", on_change=_saver("language", "language_widget"),
                 label_visibility="collapsed", format_func=lambda v: "🌐 " + v)
        st.markdown(f"**{text['theme']}**")
        st.radio("theme_choice", themes, index=themes.index(saved["theme"]),
                 key=f"theme_widget_{lang}", on_change=_saver("theme", f"theme_widget_{lang}"),
                 label_visibility="collapsed", format_func=lambda v: text["theme_" + v])
        st.markdown(f"**{text['text_size']}**")
        st.radio("size_choice", sizes, index=sizes.index(saved["text_size"]),
                 key=f"size_widget_{lang}", on_change=_saver("text_size", f"size_widget_{lang}"),
                 label_visibility="collapsed", format_func=lambda v: text["size_" + v])


def risk_style(percent):
    """Colours that follow the probability: green (low) through amber to red (high)."""
    p = min(max(percent, 0.0), 100.0)
    hue = 140.0 - 100.0 * (p / 50.0) if p <= 50.0 else 40.0 - 40.0 * ((p - 50.0) / 50.0)
    hue2 = max(hue - 25.0, -5.0)
    background = (f"linear-gradient(120deg, hsla({hue:.0f},75%,45%,0.34), "
                  f"hsla({hue2:.0f},75%,45%,0.10))")
    border = f"hsl({hue:.0f},70%,42%)"
    return background, border


def show_result(model, text, age, bmi, testosterone, follicle_count, menstrual, warnings):
    started = time.perf_counter()
    probability, agreement = predict(model, age, bmi, testosterone, follicle_count, menstrual)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    positive, uncertain = describe(probability)

    st.info(f"{text['bmi']}: {bmi:.1f} kg/m²")
    notes = []
    if warnings:
        notes.append(text["train_all"].format(items="; ".join(warnings)))
    if uncertain:
        notes.append(text["uncertain"])
    if notes:
        st.warning("\n\n".join(notes))

    percent = probability * 100.0
    band = risk_band(probability)
    icon = "✅" if band == 1 else "⚠️"
    verdict_text = text["band" + str(band)]
    background, border = risk_style(percent)
    votes = text["agreement"].format(n=len(model.estimators_), pct=f"{agreement * 100:.0f}")
    if band == 2:
        votes = (text["lean_pos"] if positive else text["lean_neg"]) + "<br>" + votes
    marker = min(max(percent, 1.0), 99.0)

    st.markdown(
        f'<div class="pcos-result">'
        f'<div class="pcos-verdict" style="background:{background};border:1px solid {border}">'
        f'{icon} {verdict_text}<small>{votes}</small></div>'
        f'<div class="pcos-box"><span>{text["estimate"]}</span>'
        f'<div class="pcos-prob">{percent:.1f}%</div>'
        f'<div class="pcos-gauge"><div class="pcos-tick" style="left:'
        f'{DECISION_THRESHOLD * 100:.0f}%"></div>'
        f'<div class="pcos-marker" style="left:{marker:.1f}%"></div></div>'
        f'<p>{text["threshold_note"]}</p></div></div>'
        f'<div class="pcos-tiles">'
        f'<div class="pcos-tile"><span>{text["tile_model"]}</span><b>Random Forest</b></div>'
        f'<div class="pcos-tile"><span>{text["tile_time"]}</span><b>{elapsed_ms:.1f} ms</b></div>'
        f'<div class="pcos-tile"><span>{text["tile_features"]}</span>'
        f'<b>{len(HYBRID_FEATURES)}</b></div></div>',
        unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="PCOS Risk Screening", page_icon="🩺", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(STYLE, unsafe_allow_html=True)

    lang = LANGUAGES.get(prefs()["language"], "en")
    text = TEXT[lang]
    apply_display_settings()
    sidebar(text, lang)
    model = load_model()

    st.markdown(
        f'<div class="pcos-hero"><h1>🩺 {TEXT["en"]["title"]}</h1><p>{text["subtitle"]}</p></div>',
        unsafe_allow_html=True)

    with st.form("patient_form"):
        with st.container(border=True):
            pill(text["basic"])
            col_age, col_weight, col_height = st.columns(3)
            a, w, h = (HARD_LIMITS[k] for k in ("Age", "Weight", "Height"))
            age = col_age.number_input(
                text["age"], value=None, min_value=int(a[0]), max_value=int(a[1]), step=1,
                format="%d", placeholder=f"{a[0]:g}–{a[1]:g}", key="age")
            weight = col_weight.number_input(
                text["weight"], value=None, min_value=w[0], max_value=w[1], step=0.1,
                format="%.1f", placeholder=f"{w[0]:g}–{w[1]:g}", key="weight")
            height = col_height.number_input(
                text["height"], value=None, min_value=h[0], max_value=h[1], step=0.1,
                format="%.1f", placeholder=f"{h[0]:g}–{h[1]:g}", key="height")
        with st.container(border=True):
            pill(text["clinical"])
            col_cycle, col_testosterone, col_follicles = st.columns(3)
            menstrual = col_cycle.selectbox(
                text["menstrual"], list(MENSTRUAL_CHOICES), index=None,
                format_func=lambda key: text[key], placeholder=text["select"], key="menstrual")
            t = HARD_LIMITS["Testosterone"]
            f = HARD_LIMITS["Antral follicle count"]
            testosterone = col_testosterone.number_input(
                text["testosterone"], value=None, min_value=t[0], max_value=t[1], step=0.1,
                format="%.1f", placeholder=f"{t[0]:g}–{t[1]:g}", key="testosterone")
            follicle_count = col_follicles.number_input(
                text["afc"], value=None, min_value=int(f[0]), max_value=int(f[1]), step=1,
                format="%d", placeholder=f"{f[0]:g}–{f[1]:g}", key="follicles")
        submitted = st.form_submit_button(text["predict"], use_container_width=True)

    entered = dict(age=age, weight=weight, height=height, testosterone=testosterone,
                   follicle_count=follicle_count, menstrual=menstrual)
    if submitted:
        st.session_state["last_inputs"] = entered
    payload = st.session_state.get("last_inputs")

    if payload is None:
        st.markdown(
            f'<div class="pcos-waiting"><div class="ico">🩺</div>'
            f'<h3>{text["wait_title"]}</h3><p>{text["waiting"]}</p>'
            f'<div class="pcos-steps"><span class="pcos-step"><i>1</i>{text["step1"]}</span>'
            f'<span class="pcos-arrow">→</span>'
            f'<span class="pcos-step"><i>2</i>{text["step2"]}</span>'
            f'<span class="pcos-arrow">→</span>'
            f'<span class="pcos-step"><i>3</i>{text["step3"]}</span></div></div>',
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
