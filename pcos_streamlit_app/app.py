"""PCOS risk screening prototype (Random Forest + Hybrid feature selection).

Research prototype. It is not a diagnostic tool.
"""

from io import BytesIO
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
        "tab_pred": "Prediction", "tab_csv": "CSV upload", "tab_report": "Download report",
        "band_short": ("Low", "Medium", "High"),
        "xai_title": "Why this result",
        "xai_up": "These inputs increased the estimate the most. A longer bar means a bigger effect.",
        "xai_down": "These inputs lowered the estimate the most. A longer bar means a bigger effect.",
        "xai_value": "Your value: {value}",
        "impact": ("High impact", "Medium impact", "Low impact"),
        "xai_main_one": "Your <b>{a}</b> contributed most to this result.",
        "xai_main_two": "Your <b>{a}</b> and <b>{b}</b> contributed most to this result.",
        "xai_note": "This shows what the model used, not a medical cause.",
        "why_unavailable": "The explanation could not be generated for this input.",
        "input_names": {"Age": "Age", "BMI": "BMI", "Testosterone": "Testosterone",
                        "Antral follicle count": "Antral follicle count",
                        "Menstrual cycle": "Menstrual cycle"},
        "rec_title": "What to do next",
        "rec_do_title": "What you can do", "rec_avoid_title": "What to avoid",
        "rec_do": (
            ("Maintain healthy eating and at least 150 minutes of moderate physical activity per "
             "week; include muscle strengthening and limit sedentary time. [2]",
             "A low estimate does not rule out PCOS. Consult a clinician if you have irregular or "
             "absent periods or signs of high androgens (for example excess hair or acne). [1]",
             "Seek evaluation for irregular periods, as they can also signal endometrial "
             "problems. [1]"),
            ("Arrange a clinical assessment; the model is uncertain in this range. [1]",
             "Early care helps protect health and well-being. [1]",
             "PCOS is diagnosed only after other causes are excluded, and needs at least two of: "
             "signs of high androgens or raised testosterone, irregular or absent periods, "
             "polycystic ovaries on ultrasound. [1]",
             "Record your cycle pattern, symptoms and family history (PCOS or type 2 diabetes) "
             "to share with the clinician. [1]",
             "Maintain healthy eating and at least 150 minutes of moderate activity per week. "
             "[1][2]"),
            ("See a clinician promptly; early care helps protect health and well-being. [1]",
             "PCOS is confirmed only after clinical assessment (at least two of: signs of high "
             "androgens or raised testosterone, irregular or absent periods, polycystic ovaries "
             "on ultrasound), once other causes are excluded. [1]",
             "If PCOS is confirmed, ask about blood tests for insulin resistance and "
             "cardiovascular risk. [1]",
             "Discuss treatment options with the clinician through shared decision-making. [1]",
             "Maintain healthy eating and physical activity (150+ minutes per week); they matter "
             "even without weight loss. [1][2]",
             "PCOS can affect mental health; seek support if you feel anxious or low. [1]")),
        "rec_avoid": (
            ("Do not interpret this result as confirmation that PCOS is absent.",
             "Do not start or stop any medication without medical advice."),
            ("Do not self-diagnose or rely on this result alone.",
             "Do not assume ovarian cysts are required for PCOS; they are not. [1]",
             "Do not start or stop any medication without medical advice."),
            ("Do not self-diagnose or delay seeking medical care.",
             "Do not start or stop any medication without medical advice.")),
        "rec_sources": "Sources",
        "rec_source_list": (
            ("[1]", "WHO. Polycystic ovary syndrome (fact sheet).",
             "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome"),
            ("[2]", "WHO. Physical activity (fact sheet).",
             "https://www.who.int/news-room/fact-sheets/detail/physical-activity")),
        "rec_source_short": ("WHO – PCOS", "WHO – Physical activity"),
        "rec_note": "General information, not a diagnosis.",
        "report_wait": "Press Predict first. The report can be downloaded here afterwards.",
        "report_title": "Download your report",
        "report_intro": "A one-page PDF with the result, the reasons, advice and sources.",
        "report_btn": "Download report (PDF)",
        "report_fallback": "The Bangla report could not be created here, so the report is in "
                           "English.",
        "rp_title": "PCOS Risk Screening Report",
        "rp_sub": "Research prototype · Generated {when}",
        "rp_inputs": "Information entered",
        "rp_bmi": "BMI (calculated)",
        "rp_why": "Why the model gave this result",
        "rp_why_note": "How much each input pushed the estimate. Raises = towards PCOS, Lowers = "
                       "away from PCOS. Points are percentage points. This shows what the model "
                       "used, not a medical cause.",
        "rp_cols": ("Input", "Your value", "Effect", "Points"),
        "rp_up": "Raises", "rp_down": "Lowers",
        "rp_verdict": "Estimated probability {pct}% · decision threshold 50%",
        "rp_foot": "PCOS Risk Screening: research prototype, not a diagnosis.",
        "csv_title": "Predict for many people from a CSV file",
        "csv_help": "Needed columns: Age, Weight, Height, Testosterone, Antral follicle count, "
                    "Menstrual cycle (Regular or Irregular, or 0/1 where 1 = irregular). A BMI "
                    "column can replace Weight and Height. Other columns are ignored.",
        "csv_template": "Download template CSV",
        "csv_upload": "Choose a CSV file",
        "csv_predict": "Predict all rows",
        "csv_missing": "These required columns are missing: {cols}.",
        "csv_empty": "The file has no rows.",
        "csv_toobig": "The file has {n} rows; the limit is {limit}.",
        "csv_unreadable": "The file could not be read as a CSV.",
        "csv_summary": "{ok} rows predicted, {bad} rows need checking.",
        "csv_ignored": "Ignored columns: {cols}.",
        "csv_download": "Download results (CSV)",
        "col_row": "Row", "col_prob": "PCOS probability (%)", "col_band": "Risk level",
        "col_note": "Note", "invalid": "Check input",
        "cycle_bad": "Menstrual cycle must be Regular or Irregular (or 0/1).",
        "need_columns": {"age": "Age", "weight": "Weight", "height": "Height",
                         "testosterone": "Testosterone", "afc": "Antral follicle count",
                         "menstrual": "Menstrual cycle"},
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
        "tab_pred": "পূর্বাভাস", "tab_csv": "CSV আপলোড", "tab_report": "রিপোর্ট ডাউনলোড",
        "band_short": ("কম", "মাঝারি", "বেশি"),
        "xai_title": "ফলাফল এমন হলো কেন",
        "xai_up": "এই তথ্যগুলো সম্ভাবনা সবচেয়ে বেশি বাড়িয়েছে। দাগ যত লম্বা, প্রভাব তত বেশি।",
        "xai_down": "এই তথ্যগুলো সম্ভাবনা সবচেয়ে বেশি কমিয়েছে। দাগ যত লম্বা, প্রভাব তত বেশি।",
        "xai_value": "আপনার মান: {value}",
        "impact": ("প্রভাব অনেক", "প্রভাব মোটামুটি", "প্রভাব সামান্য"),
        "xai_main_one": "আপনার <b>{a}</b> ফলাফলে সবচেয়ে বেশি প্রভাব ফেলেছে।",
        "xai_main_two": "আপনার <b>{a}</b> ও <b>{b}</b> ফলাফলে সবচেয়ে বেশি প্রভাব ফেলেছে।",
        "xai_note": "এটা মডেল কীসের উপর ভর করেছে তা দেখায়, রোগের কারণ নয়।",
        "why_unavailable": "এই তথ্যের জন্য ব্যাখ্যা তৈরি করা যায়নি।",
        "input_names": {"Age": "বয়স", "BMI": "BMI", "Testosterone": "টেস্টোস্টেরন",
                        "Antral follicle count": "অ্যান্ট্রাল ফলিকল কাউন্ট (AFC)",
                        "Menstrual cycle": "মাসিক চক্র"},
        "rec_title": "এখন কী করবেন",
        "rec_do_title": "যা করা ভালো", "rec_avoid_title": "যা করবেন না",
        "rec_do": (
            ("স্বাস্থ্যকর খাবার খান আর সপ্তাহে অন্তত ১৫০ মিনিট হালকা-মাঝারি ব্যায়াম বা হাঁটাহাঁটি "
             "করুন। পেশির ব্যায়ামও করুন, আর অনেকক্ষণ একটানা বসে থাকবেন না। [২]",
             "ফলাফল কম হলেও PCOS নেই, এমন নয়। মাসিক অনিয়মিত বা বন্ধ থাকলে, কিংবা পুরুষ-হরমোন "
             "বেশির লক্ষণ (যেমন অতিরিক্ত লোম, ব্রণ) থাকলে ডাক্তার দেখান। [১]",
             "মাসিক অনিয়মিত হলে ডাক্তারকে দেখিয়ে নিন, কারণ এটা জরায়ুর ভেতরের স্তরের সমস্যারও "
             "ইঙ্গিত হতে পারে। [১]"),
            ("ডাক্তার দেখিয়ে মূল্যায়ন করিয়ে নিন। এই ধাপে মডেল নিজেই নিশ্চিত নয়। [১]",
             "আগেভাগে চিকিৎসা নিলে শরীর ও মনের যত্ন নেওয়া সহজ হয়। [১]",
             "অন্য কারণ বাদ দিয়ে এই তিনটির অন্তত দুটি মিললে PCOS ধরা হয়: পুরুষ-হরমোন বেশির "
             "লক্ষণ (বা টেস্টোস্টেরন বেশি), অনিয়মিত বা বন্ধ মাসিক, আলট্রাসাউন্ডে ডিম্বাশয়ে ছোট ছোট "
             "থলি। [১]",
             "মাসিকের ধরন, উপসর্গ আর পরিবারে PCOS বা টাইপ-২ ডায়াবেটিস আছে কি না, ডাক্তারকে "
             "জানানোর জন্য লিখে রাখুন। [১]",
             "স্বাস্থ্যকর খাবার খান আর সপ্তাহে অন্তত ১৫০ মিনিট হালকা-মাঝারি ব্যায়াম করুন। [১][২]"),
            ("দেরি না করে একজন ডাক্তার দেখান। শুরুতে চিকিৎসা নিলে শরীরের যত্ন নেওয়া সহজ হয়। [১]",
             "PCOS আছে কি না, তা শুধু ডাক্তারই নিশ্চিত করতে পারেন। অন্য কারণ বাদ দিয়ে এই তিনটির "
             "অন্তত দুটি মিললে PCOS ধরা হয়: পুরুষ-হরমোন বেশির লক্ষণ (যেমন টেস্টোস্টেরন বেশি), "
             "অনিয়মিত বা বন্ধ মাসিক, আলট্রাসাউন্ডে ডিম্বাশয়ে ছোট ছোট থলি। [১]",
             "PCOS ধরা পড়লে ডাক্তারকে জিজ্ঞেস করুন ইনসুলিন রেজিস্ট্যান্স আর হৃদরোগের ঝুঁকি দেখতে "
             "কোন রক্তপরীক্ষা লাগবে। [১]",
             "কী ধরনের চিকিৎসা হবে, তা ডাক্তারের সঙ্গে কথা বলে আপনার পছন্দ-অপছন্দ মিলিয়ে ঠিক "
             "করুন। [১]",
             "স্বাস্থ্যকর খাবার খান আর সপ্তাহে অন্তত ১৫০ মিনিট হালকা-মাঝারি ব্যায়াম বা হাঁটাহাঁটি "
             "করুন। ওজন না কমলেও এতে উপকার হয়। [১][২]",
             "PCOS মনের উপরও চাপ ফেলতে পারে। দুশ্চিন্তা বা মন খারাপ লাগলে কাউকে বলুন, সাহায্য "
             "নিন। [১]")),
        "rec_avoid": (
            ("ফলাফল কম দেখে ধরে নেবেন না যে PCOS নেই।",
             "ডাক্তারকে না জিজ্ঞেস করে কোনো ওষুধ শুরু বা বন্ধ করবেন না।"),
            ("শুধু এই ফলাফলের উপর ভরসা করে নিজে রোগ ধরে নেবেন না।",
             "ভাববেন না যে PCOS হতে হলে ডিম্বাশয়ে সিস্ট থাকতেই হবে; এটা জরুরি নয়। [১]",
             "ডাক্তারকে না জিজ্ঞেস করে কোনো ওষুধ শুরু বা বন্ধ করবেন না।"),
            ("নিজে নিজে রোগ ধরে নেবেন না, আর ডাক্তার দেখাতে দেরি করবেন না।",
             "ডাক্তারকে না জিজ্ঞেস করে কোনো ওষুধ শুরু বা বন্ধ করবেন না।")),
        "rec_sources": "তথ্যের উৎস",
        "rec_source_list": (
            ("[১]", "WHO. Polycystic ovary syndrome (fact sheet)।",
             "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome"),
            ("[২]", "WHO. Physical activity (fact sheet)।",
             "https://www.who.int/news-room/fact-sheets/detail/physical-activity")),
        "rec_source_short": ("WHO – PCOS", "WHO – শারীরিক কার্যকলাপ"),
        "rec_note": "এটা সাধারণ তথ্য, রোগ নির্ণয় নয়।",
        "report_wait": "আগে Predict চাপুন। তারপর রিপোর্ট এখানে ডাউনলোড করতে পারবেন।",
        "report_title": "রিপোর্ট ডাউনলোড করুন",
        "report_intro": "এক পাতার PDF: ফলাফল, কারণ, পরামর্শ ও তথ্যের উৎস।",
        "report_btn": "রিপোর্ট ডাউনলোড (PDF)",
        "report_fallback": "বাংলা রিপোর্ট এখানে বানানো যায়নি, তাই রিপোর্ট ইংরেজিতে দেওয়া হলো।",
        "rp_title": "PCOS ঝুঁকি যাচাইয়ের রিপোর্ট",
        "rp_sub": "গবেষণামূলক প্রোটোটাইপ · তৈরি হয়েছে {when}",
        "rp_inputs": "দেওয়া তথ্য",
        "rp_bmi": "BMI (হিসাব করা)",
        "rp_why": "ফলাফল এমন হলো কেন",
        "rp_why_note": "কোন তথ্য ফলাফলকে কতটা প্রভাবিত করেছে তা এখানে দেখানো হয়েছে। বাড়িয়েছে = "
                       "PCOS-এর দিকে, কমিয়েছে = দূরে। পয়েন্ট মানে শতাংশ-পয়েন্ট। এটা মডেল কীসের "
                       "উপর ভর করেছে তা বোঝায়, রোগের কারণ নয়।",
        "rp_cols": ("তথ্য", "আপনার মান", "প্রভাব", "পয়েন্ট"),
        "rp_up": "বাড়িয়েছে", "rp_down": "কমিয়েছে",
        "rp_verdict": "সম্ভাবনা {pct}% · সীমারেখা ৫০%",
        "rp_foot": "PCOS ঝুঁকি যাচাই: গবেষণামূলক প্রোটোটাইপ, রোগ নির্ণয় নয়।",
        "csv_title": "CSV ফাইল থেকে অনেক জনের পূর্বাভাস",
        "csv_help": "দরকারি column: Age, Weight, Height, Testosterone, Antral follicle count, "
                    "Menstrual cycle (Regular বা Irregular, অথবা 0/1 যেখানে 1 = অনিয়মিত)। "
                    "Weight ও Height-এর বদলে BMI column দিলেও চলবে। অন্য column বাদ যাবে।",
        "csv_template": "টেমপ্লেট CSV ডাউনলোড",
        "csv_upload": "CSV ফাইল বাছুন",
        "csv_predict": "সব row-এর পূর্বাভাস দিন",
        "csv_missing": "এই দরকারি column গুলো নেই: {cols}।",
        "csv_empty": "ফাইলে কোনো row নেই।",
        "csv_toobig": "ফাইলে {n}টা row আছে; সীমা {limit}।",
        "csv_unreadable": "ফাইলটা CSV হিসেবে পড়া যায়নি।",
        "csv_summary": "{ok}টা row-এর পূর্বাভাস হয়েছে, {bad}টা row যাচাই করতে হবে।",
        "csv_ignored": "বাদ দেওয়া column: {cols}।",
        "csv_download": "ফলাফল ডাউনলোড (CSV)",
        "col_row": "Row", "col_prob": "PCOS সম্ভাব্যতা (%)", "col_band": "ঝুঁকির স্তর",
        "col_note": "মন্তব্য", "invalid": "তথ্য যাচাই করুন",
        "cycle_bad": "মাসিক চক্র অবশ্যই নিয়মিত বা অনিয়মিত (অথবা 0/1) হতে হবে।",
        "need_columns": {"age": "Age", "weight": "Weight", "height": "Height",
                         "testosterone": "Testosterone", "afc": "Antral follicle count",
                         "menstrual": "Menstrual cycle"},
        "agreement": "মডেলের {n}টি সিদ্ধান্ত-গাছের {pct}%-ই একই ফলাফল দিয়েছে।",
        "uncertain": "ফলাফল সিদ্ধান্ত সীমার কাছাকাছি, তাই অনিশ্চিত। ক্লিনিক্যাল মূল্যায়ন প্রয়োজন।",
        "why_title": "মডেল সম্পর্কে",
        "why_body": "Hybrid feature selection দিয়ে বাছাই করা ১০টি feature-এর উপর Tuned Random "
                    "Forest। প্রশিক্ষণে SMOTE শুধু training অংশে ব্যবহার হয়েছে। আলাদা করে রাখা "
                    "টেস্ট সেটে (৬০০ রেকর্ড, একবারই ব্যবহৃত): accuracy 0.983, F1 0.959, "
                    "ROC-AUC 0.999।",
        "data_title": "ডেটা",
        "data_body": "দুটি পাবলিক ডেটাসেট থেকে ৪,০০০ রেকর্ড। টেস্ট সেট খোলার আগেই মডেল চূড়ান্ত "
                     "করা হয়েছিল।",
        "disclaimer_title": "সতর্কতা",
        "disclaimer_body": "এটি একটি গবেষণামূলক প্রোটোটাইপ, এখনো ডাক্তারি পরীক্ষায় যাচাই হয়নি। এর "
                           "ফল দেখে PCOS আছে বা নেই—এমন সিদ্ধান্ত নেবেন না। সন্দেহ থাকলে ডাক্তারের "
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
    color: #ffffff; padding: 12px 20px; border-radius: 14px; margin-bottom: 4px;
}
.pcos-hero h1 { color: #ffffff; margin: 0 0 2px 0; padding: 0; font-size: 1.4rem; }
.pcos-hero p { margin: 0; opacity: 1; font-size: 0.88rem; }
.pcos-pill {
    display: inline-block; padding: 3px 14px; border-radius: 999px; margin-bottom: 6px;
    background: linear-gradient(120deg, #4f46e5, #9333ea); color: #ffffff;
    font-weight: 600; font-size: 0.9rem;
}
.pcos-box {
    border-radius: 14px; padding: 12px 14px; margin-bottom: 8px;
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
.pcos-error-list { display: flex; flex-direction: column; gap: 10px; margin: 4px 0 12px 0; }
.pcos-error { display: flex; align-items: center; gap: 12px; padding: 13px 16px; margin: 0;
    border-radius: 12px; border: 1px solid #f87171; border-left: 6px solid #dc2626;
    background: linear-gradient(100deg, #fecaca, #fee2e2); font-weight: 600;
    box-shadow: 0 1px 4px rgba(220, 38, 38, 0.25); }
.pcos-error, .pcos-error span { color: #7f1d1d !important; }
.pcos-error .pcos-error-icon { color: #ffffff !important; }
.pcos-error-icon { flex: none; width: 24px; height: 24px; border-radius: 50%; background: #dc2626;
    color: #ffffff !important; font-weight: 800; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.95rem; }
.pcos-gr { margin: 8px 0; }
.pcos-gl { display: flex; align-items: center; gap: 10px; }
.pcos-gl b { font-size: 0.9rem; }
.pcos-gt { font-size: 0.84rem; line-height: 1.3; margin-top: 2px; }
.pcos-swatch { width: 34px; height: 16px; border-radius: 5px; flex: none; }
.pcos-rcard { border-radius: 14px; padding: 11px 18px 10px 18px; text-align: center;
    margin: 6px 0 10px 0; box-sizing: border-box; }
.pcos-rcard .v { font-size: 1.15rem; font-weight: 700; }
.pcos-rcard .e { font-size: 0.88rem; }
.pcos-rcard .p { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 2px 0; }
.pcos-rcard .n { font-size: 0.8rem; }
.pcos-rcard.b1, .pcos-rcard.b1 *, .pcos-rcard.b2, .pcos-rcard.b2 * { color: #1f2937 !important; }
.pcos-rcard.b3, .pcos-rcard.b3 * { color: #ffffff !important; }
.pcos-gauge { position: relative; height: 14px; border-radius: 999px; margin: 12px 6px 8px 6px;
    background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #e11d48 100%); }
.pcos-marker { position: absolute; top: -5px; width: 6px; height: 24px; border-radius: 3px;
    background: #111827; border: 2px solid #ffffff; transform: translateX(-50%); }
.pcos-tick { position: absolute; top: -3px; width: 2px; height: 20px;
    background: rgba(255, 255, 255, 0.95); transform: translateX(-50%); }
.pcos-rec { padding: 10px 14px; }
.pcos-rec h4 { margin: 0 0 6px 0; font-size: 1rem; }
.pcos-why .wh { font-weight: 700; font-size: 0.95rem; }
.pcos-legend { font-size: 0.8rem; opacity: 0.9; margin: 1px 0 4px 0; }
.pcos-xr { display: grid; grid-template-columns: minmax(0, 11rem) 8rem 1fr; gap: 8px;
    align-items: center; margin: 5px 0; font-size: 0.86rem; line-height: 1.2; }
.pcos-xr small { font-size: 0.76rem; opacity: 0.8; }
.pcos-xr .tag { display: inline-block; text-align: center; border-radius: 999px; padding: 1px 8px;
    font-size: 0.74rem; font-weight: 700; color: #ffffff !important; }
.pcos-xr .tag.up { background: #e11d48; }
.pcos-xr .tag.down { background: #059669; }
.pcos-xtrack { position: relative; height: 14px; border-radius: 7px;
    background: rgba(127,127,127,0.18); }
.pcos-xbar { position: absolute; top: 0; height: 14px; border-radius: 7px; }
.pcos-xbar.pos { background: #e11d48; }
.pcos-xbar.neg { background: #059669; }
.pcos-why .sum { font-size: 0.88rem; margin-top: 4px; }
.pcos-why .sum .note { font-size: 0.76rem; opacity: 0.85; }
.pcos-rec-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px;
    padding-top: 8px; border-top: 1px solid rgba(127,127,127,0.35); }
.pcos-rec h5 { margin: 0 0 2px 0; font-size: 0.92rem; }
.pcos-rec ol { margin: 0; padding-left: 1.3rem; }
.pcos-rec li { font-size: 0.85rem; line-height: 1.35; margin-bottom: 2px; }
.pcos-rec .do h5 { color: #059669; }
.pcos-rec .avoid h5 { color: #e11d48; }
.pcos-rec .src { font-size: 0.78rem; margin-top: 6px; line-height: 1.45; opacity: 0.95; }
.pcos-rec .src a { color: #6366f1; text-decoration: underline; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"], .block-container { padding: 3rem 0.8rem 1rem 0.8rem; }
    .pcos-hero { padding: 12px 14px; }
    .pcos-hero h1 { font-size: 1.3rem; }
    .pcos-hero p { font-size: 0.85rem; }
    .pcos-rec-cols { grid-template-columns: 1fr; }
    .pcos-xr { grid-template-columns: 1fr auto; }
    .pcos-xr .pcos-xtrack { grid-column: 1 / -1; }
}
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
[data-testid="stMainBlockContainer"], .block-container {
    padding: 2.6rem 1.2rem 0.8rem 1rem; max-width: 740px; margin-left: 0; margin-right: auto; }
[data-testid="stVerticalBlock"] { gap: 0.45rem; }
[data-testid="stTabs"] button[role="tab"] { font-weight: 600; padding: 0.35rem 0.9rem; }
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


def validate_inputs(age, weight, height, testosterone, follicle_count, menstrual, lang="en",
                    bmi_value=None):
    """Return (errors, warnings, bmi). Empty fields and impossible values are
    errors; values outside the training range are warnings. When bmi_value is
    given (CSV files with a BMI column), weight and height are not required."""
    text = TEXT[lang]
    names = text["fields"]
    units = text["units"]
    errors, warnings = [], []
    use_bmi = bmi_value is not None

    fields = [("Age", age)]
    if not use_bmi:
        fields += [("Weight", weight), ("Height", height)]
    fields += [("Testosterone", testosterone), ("Antral follicle count", follicle_count)]
    for label, value in fields:
        if value is None:
            errors.append(text["required"].format(field=names[label]))
    if menstrual not in MENSTRUAL_CHOICES:
        errors.append(text["required"].format(field=names["Menstrual cycle"]))
    if errors:
        return errors, warnings, None

    for label, value in fields:
        if label in ("Age", "Weight", "Height") and value == 0:
            errors.append(text["zero"].format(field=names[label]))
    for label, value in fields:
        if not np.isfinite(value):
            errors.append(text["finite"].format(field=names[label]))
    if use_bmi and not np.isfinite(bmi_value):
        errors.append(text["finite"].format(field=names["BMI"]))
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

    bmi = float(bmi_value) if use_bmi else compute_bmi(weight, height)
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


# ----------------------------------------------------- explanation (XAI)

# Which original inputs each model feature is built from.
FEATURE_INPUTS = {
    "Menstrual_Irregularity": ["Menstrual cycle"],
    "Testosterone_Levelng/dL_Antral_Follicle_Count": ["Testosterone", "Antral follicle count"],
    "Antral_Follicle_Count_Squared": ["Antral follicle count"],
    "BMI_Antral_Follicle_Count": ["BMI", "Antral follicle count"],
    "Antral_Follicle_Count": ["Antral follicle count"],
    "BMI_Testosterone_Levelng/dL": ["BMI", "Testosterone"],
    "Testosterone_Levelng/dL": ["Testosterone"],
    "Testosterone_Levelng/dL_Squared": ["Testosterone"],
    "Age_Antral_Follicle_Count": ["Age", "Antral follicle count"],
    "Age": ["Age"],
}
INPUT_ORDER = ["Age", "BMI", "Testosterone", "Antral follicle count", "Menstrual cycle"]


def positive_class_values(raw, n_features):
    """Return the SHAP values of the PCOS class for one row, whatever shape the
    installed shap version returns."""
    if isinstance(raw, list):
        values = np.asarray(raw[1])[0]
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3:
            values = raw[0, :, 1]
        elif raw.ndim == 2:
            values = raw[0]
        else:
            values = raw
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != n_features:
        raise ValueError("Unexpected SHAP output shape.")
    return values


def aggregate_by_input(by_feature):
    """Share each model feature's contribution equally between the inputs it is
    built from. The total is unchanged."""
    totals = {name: 0.0 for name in INPUT_ORDER}
    for feature, value in by_feature.items():
        parts = FEATURE_INPUTS[feature]
        for part in parts:
            totals[part] += value / len(parts)
    return totals


@st.cache_resource(show_spinner=False)
def load_explainer():
    import shap
    return shap.TreeExplainer(load_model())


@st.cache_data(show_spinner=False)
def explain(age, bmi, testosterone, follicle_count, menstrual):
    """Contribution of every model feature, in percentage points of the PCOS
    probability. Returns None if the explanation cannot be produced."""
    try:
        features = build_features(age, bmi, testosterone, follicle_count,
                                  MENSTRUAL_CHOICES[menstrual])
        frame = model_input(features)
        raw = load_explainer().shap_values(frame, check_additivity=False)
        values = positive_class_values(raw, len(HYBRID_FEATURES))
        return {name: float(v) * 100.0 for name, v in zip(HYBRID_FEATURES, values)}
    except Exception:
        return None


# ------------------------------------------------- impact of each input

def impact_level(ratio):
    """0 = high, 1 = medium, 2 = low impact (relative to the strongest input)."""
    return 0 if ratio >= 0.6 else 1 if ratio >= 0.3 else 2


def factor_list(by_input):
    """Every input that moved the estimate, strongest first."""
    peak = max([abs(v) for v in by_input.values()] + [1e-9])
    items = []
    for name, value in by_input.items():
        if abs(value) < 0.05:
            continue
        ratio = abs(value) / peak
        items.append({"name": name, "value": value, "ratio": ratio,
                      "level": impact_level(ratio)})
    items.sort(key=lambda d: -abs(d["value"]))
    return items


def shown_factors(by_input, band):
    """Medium/High: only the inputs that raised the estimate. Low: only the inputs
    that lowered it."""
    want_up = band >= 2
    return [d for d in factor_list(by_input) if (d["value"] > 0) == want_up]


def input_values(text, payload, bmi):
    return {"Age": f'{payload["age"]:g} {text["units"]["years"]}',
            "BMI": f"{bmi:.1f}",
            "Testosterone": f'{payload["testosterone"]:g} ng/dL',
            "Antral follicle count": f'{payload["follicle_count"]:g}',
            "Menstrual cycle": text[payload["menstrual"]]}


def xai_html(text, payload, result, by_input):
    """The 'Why this result' block shown inside the recommendation box."""
    if not by_input:
        return f'<div class="pcos-legend">{text["why_unavailable"]}</div>'
    band = result["band"]
    chosen = shown_factors(by_input, band)
    if not chosen:
        return ""
    values = input_values(text, payload, result["bmi"])
    rows = ""
    for item in chosen:
        up = item["value"] > 0
        width = max(item["ratio"] * 100.0, 4.0)
        rows += (f'<div class="pcos-xr"><span class="lab"><b>{text["input_names"][item["name"]]}'
                 f'</b><br><small>{text["xai_value"].format(value=values[item["name"]])}</small>'
                 f'</span><span class="tag {"up" if up else "down"}">'
                 f'{text["impact"][item["level"]]}</span><div class="pcos-xtrack">'
                 f'<div class="pcos-xbar {"pos" if up else "neg"}" style="left:0;'
                 f'width:{width:.0f}%"></div></div></div>')
    names = [text["input_names"][item["name"]] for item in chosen[:2]]
    if len(names) == 2:
        main = text["xai_main_two"].format(a=names[0], b=names[1])
    else:
        main = text["xai_main_one"].format(a=names[0])
    legend = text["xai_up"] if band >= 2 else text["xai_down"]
    return (f'<div class="pcos-why"><div class="wh">🔍 {text["xai_title"]}</div>'
            f'<div class="pcos-legend">{legend}</div>{rows}'
            f'<div class="sum">{main} <span class="note">{text["xai_note"]}</span></div></div>')


# ------------------------------------------------------- recommendations

def recommendation_html(text, band, why_html="", lang="en"):
    index = band - 1
    style = ' style="list-style-type:bengali"' if lang == "bn" else ""
    do_items = "".join(f"<li>{item}</li>" for item in text["rec_do"][index])
    avoid_items = "".join(f"<li>{item}</li>" for item in text["rec_avoid"][index])
    sources = " &nbsp;·&nbsp; ".join(
        f'{tag} <a href="{url}" target="_blank" rel="noopener">{short}</a>'
        for (tag, _title, url), short in zip(text["rec_source_list"], text["rec_source_short"]))
    return (f'<div class="pcos-box accent pcos-rec"><h4>🩺 {text["rec_title"]}</h4>{why_html}'
            f'<div class="pcos-rec-cols"><div class="do"><h5>✅ {text["rec_do_title"]}</h5>'
            f'<ol{style}>{do_items}</ol></div><div class="avoid"><h5>🚫 {text["rec_avoid_title"]}'
            f'</h5><ol{style}>{avoid_items}</ol></div></div>'
            f'<div class="src"><b>{text["rec_sources"]}:</b> {sources}. {text["rec_note"]}</div>'
            f'</div>')


# ------------------------------------------------------------ PDF report

REPORT_COLOURS = {1: "#2e8b3a", 2: "#c98a1a", 3: "#b91c1c"}
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
APP_DIR = Path(__file__).resolve().parent


def report_rows(text, payload, result, by_input):
    """Rows for the 'why' table of the report (every input)."""
    if not by_input:
        return []
    values = input_values(text, payload, result["bmi"])
    rows = []
    for item in factor_list(by_input):
        word = text["rp_up"] if item["value"] > 0 else text["rp_down"]
        rows.append((text["input_names"][item["name"]], values[item["name"]],
                     f'{word} · {text["impact"][item["level"]]}', f'{item["value"]:+.1f}',
                     item["value"] > 0))
    return rows


def report_html(text, lang, payload, result, by_input):
    """One-page report as HTML (turned into a PDF by WeasyPrint)."""
    from datetime import datetime, timezone
    from html import escape

    def L(value):
        value = str(value)
        return value.translate(BN_DIGITS) if lang == "bn" else value

    def E(value):
        return escape(L(value))

    band = result["band"]
    colour = REPORT_COLOURS[band]
    when = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    fonts = sorted(APP_DIR.glob("*Bengali*.ttf")) + sorted((APP_DIR / "fonts").glob("*Bengali*.ttf"))
    font_face = ""
    family = "'Noto Sans Bengali','Hind Siliguri','Lohit Bengali','Noto Sans','DejaVu Sans',sans-serif"
    if fonts:
        font_face = (f"@font-face{{font-family:'PCOSBengali';src:url('{fonts[0].as_uri()}');}}")
        family = "'PCOSBengali'," + family
    names = text["fields"]
    left = [("Age", f'{payload["age"]:g} {text["units"]["years"]}'),
            ("Weight", f'{payload["weight"]:g} kg'), ("Height", f'{payload["height"]:g} cm')]
    right = [(text["rp_bmi"], f'{result["bmi"]:.1f}'),
             (names["Testosterone"], f'{payload["testosterone"]:g} ng/dL'),
             (names["Antral follicle count"], f'{payload["follicle_count"]:g}'),
             (names["Menstrual cycle"], text[payload["menstrual"]])]
    left = [(names[key], value) for key, value in left]

    def grid(rows):
        return ('<table class="dt">' + "".join(
            f"<tr><td>{E(a)}</td><td>{E(b)}</td></tr>" for a, b in rows) + "</table>")

    notes = [text["agreement"].format(n=result["n_trees"],
                                      pct=f"{result['agreement'] * 100:.0f}")]
    if band == 2:
        notes.append(text["uncertain"])
    if result["warnings"]:
        notes.append(text["train_all"].format(items="; ".join(result["warnings"])))
    why = ""
    rows = report_rows(text, payload, result, by_input)
    if rows:
        head = "".join(f"<th>{E(c)}</th>" for c in text["rp_cols"])
        body = "".join(
            f'<tr><td>{E(a)}</td><td>{E(b)}</td><td class="{"up" if up else "dn"}">{E(c)}</td>'
            f'<td>{E(d)}</td></tr>' for a, b, c, d, up in rows)
        why = (f'<h2>{E(text["rp_why"])}</h2><div class="sm">{E(text["rp_why_note"])}</div>'
               f'<table class="dt wide"><tr>{head}</tr>{body}</table>')
    ol = ' style="list-style-type:bengali"' if lang == "bn" else ""
    do_items = "".join(f"<li>{E(x)}</li>" for x in text["rec_do"][band - 1])
    avoid_items = "".join(f"<li>{E(x)}</li>" for x in text["rec_avoid"][band - 1])
    sources = "<br>".join(
        f'{E(tag)} {escape(title)} {escape(url)}' for tag, title, url in text["rec_source_list"])
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>
{font_face}
@page {{ size: A4; margin: 11mm 13mm 13mm 13mm; }}
body {{ font-family: {family}; font-size: 9.4pt; line-height: 1.45; color: #111; margin: 0; }}
.hd {{ background: #4338ca; color: #fff; padding: 9px 14px; border-radius: 8px; }}
.hd h1 {{ margin: 0; font-size: 16pt; }} .hd div {{ font-size: 8.5pt; }}
.res {{ background: {colour}; color: #fff; border-radius: 8px; padding: 8px 14px; margin: 7px 0 3px 0;
        font-weight: bold; }}
.res span.r {{ float: right; }}
.sm {{ font-size: 8pt; color: #475569; }}
h2 {{ font-size: 11pt; color: #312e81; margin: 9px 0 3px 0; border-bottom: 1px solid #c7d2fe;
      padding-bottom: 2px; }}
table.dt {{ border-collapse: collapse; width: 100%; }}
table.dt td, table.dt th {{ border: 1px solid #cbd5e1; padding: 3px 7px; text-align: left;
                           font-size: 9pt; }}
table.dt th {{ background: #eef2ff; }}
table.wide {{ margin-top: 3px; }}
.up {{ color: #be123c; font-weight: bold; }} .dn {{ color: #047857; font-weight: bold; }}
table.cols {{ width: 100%; border-collapse: collapse; }}
table.cols td {{ vertical-align: top; width: 50%; padding: 0 8px 0 0; border: none; }}
h3 {{ margin: 0 0 2px 0; font-size: 10pt; }}
ol {{ margin: 0; padding-left: 16px; }} li {{ margin-bottom: 2px; }}
.src {{ font-size: 7.8pt; color: #475569; }}
.dis {{ border: 1px solid #d97706; background: #fffbeb; border-radius: 6px; padding: 5px 9px;
        margin-top: 8px; font-size: 8.6pt; }}
.ft {{ font-size: 7.5pt; color: #64748b; margin-top: 6px; }}
</style></head><body>
<div class="hd"><h1>{E(text["rp_title"])}</h1><div>{E(text["rp_sub"].format(when=when))}</div></div>
<div class="res"><span class="r">{E(text["rp_verdict"].format(pct=f"{result['percent']:.1f}"))}</span>{E(text["band" + str(band)])}</div>
<div class="sm">{E(" ".join(notes))}</div>
<h2>{E(text["rp_inputs"])}</h2>
<table class="cols"><tr><td>{grid(left)}</td><td>{grid(right)}</td></tr></table>
{why}
<h2>{E(text["rec_title"])}</h2>
<table class="cols"><tr><td><h3 style="color:#047857">{E(text["rec_do_title"])}</h3><ol{ol}>{do_items}</ol></td>
<td><h3 style="color:#be123c">{E(text["rec_avoid_title"])}</h3><ol{ol}>{avoid_items}</ol></td></tr></table>
<h2>{E(text["rec_sources"])}</h2>
<div class="src">{sources}</div>
<div class="dis"><b>{E(text["disclaimer_title"])}:</b> {E(text["disclaimer_body"])}</div>
<div class="ft">{E(text["rp_foot"])}</div>
</body></html>"""


def build_report_basic(payload, result, by_input):
    """English one-page fallback (ReportLab) used if the HTML route is unavailable."""
    from datetime import datetime, timezone
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    text = TEXT["en"]
    band = result["band"]
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle("body", parent=base, fontName="Helvetica", fontSize=8.8, leading=11.4)
    small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=9.6,
                           textColor=colors.HexColor("#475569"))
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                        textColor=colors.HexColor("#312e81"), spaceBefore=6, spaceAfter=2,
                        keepWithNext=1)
    title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=16,
                           leading=19, textColor=colors.white)
    sub = ParagraphStyle("sub", parent=body, fontSize=8.5, textColor=colors.white)
    item = ParagraphStyle("item", parent=body, leftIndent=13, spaceAfter=1.5)

    def clean(value):
        return escape(str(value))

    def numbered(items):
        return [Paragraph(clean(x), item, bulletText=f"{i}.") for i, x in enumerate(items, 1)]

    def grid(rows, widths, head=False):
        table = Table([[Paragraph(clean(c), body) for c in row] for row in rows], colWidths=widths)
        style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                 ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                 ("BACKGROUND", (0, 0), (-1, 0 if head else -1), colors.HexColor("#eef2ff"))]
        if not head:
            style[-1] = ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff"))
        table.setStyle(TableStyle(style))
        return table

    when = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    story = []
    header = Table([[Paragraph(clean(text["rp_title"]), title)],
                    [Paragraph(clean(text["rp_sub"].format(when=when)), sub)]],
                   colWidths=[178 * mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#4338ca")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
    verdict = Table(
        [[Paragraph(f'<font color="white"><b>{clean(text["band" + str(band)])}</b></font>', body),
          Paragraph('<font color="white"><b>' + clean(text["rp_verdict"].format(
              pct=f'{result["percent"]:.1f}')) + "</b></font>", body)]],
        colWidths=[84 * mm, 94 * mm])
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(REPORT_COLOURS[band])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [header, Spacer(1, 5), verdict, Spacer(1, 3)]
    notes = [text["agreement"].format(n=result["n_trees"], pct=f"{result['agreement'] * 100:.0f}")]
    if band == 2:
        notes.append(text["uncertain"])
    if result["warnings"]:
        notes.append(text["train_all"].format(items="; ".join(result["warnings"])))
    story.append(Paragraph(clean(" ".join(notes)), small))

    story.append(Paragraph(clean(text["rp_inputs"]), h2))
    names = text["fields"]
    cells = [(names["Age"], f'{payload["age"]:g} years'), (text["rp_bmi"], f'{result["bmi"]:.1f}'),
             (names["Weight"], f'{payload["weight"]:g} kg'),
             (names["Testosterone"], f'{payload["testosterone"]:g} ng/dL'),
             (names["Height"], f'{payload["height"]:g} cm'),
             (names["Antral follicle count"], f'{payload["follicle_count"]:g}'),
             (names["Menstrual cycle"], text[payload["menstrual"]]), ("", "")]
    story.append(grid([[cells[i][0], cells[i][1], cells[i + 1][0], cells[i + 1][1]]
                       for i in range(0, 8, 2)], [38 * mm, 51 * mm, 38 * mm, 51 * mm]))

    rows = report_rows(text, payload, result, by_input)
    if rows:
        story.append(Paragraph(clean(text["rp_why"]), h2))
        story.append(Paragraph(clean(text["rp_why_note"]), small))
        story += [Spacer(1, 2), grid([list(text["rp_cols"])] + [[a, b, c, d] for a, b, c, d, _ in rows],
                                     [52 * mm, 36 * mm, 60 * mm, 30 * mm], head=True)]

    story.append(Paragraph(clean(text["rec_title"]), h2))
    story.append(Paragraph(f"<b>{clean(text['rec_do_title'])}</b>", body))
    story += numbered(text["rec_do"][band - 1])
    story += [Spacer(1, 2), Paragraph(f"<b>{clean(text['rec_avoid_title'])}</b>", body)]
    story += numbered(text["rec_avoid"][band - 1])
    story.append(Paragraph(clean(text["rec_sources"]), h2))
    for tag, name, url in text["rec_source_list"]:
        story.append(Paragraph(f"{clean(tag)} {clean(name)} {clean(url)}", small))
    story.append(Spacer(1, 5))
    box = Table([[Paragraph(f"<b>{clean(text['disclaimer_title'])}.</b> "
                            + clean(text["disclaimer_body"]), body)]], colWidths=[178 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d97706")),
                             ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 4),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(box)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 8 * mm, text["rp_foot"])
        canvas.restoreState()

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                      topMargin=10 * mm, bottomMargin=14 * mm,
                      title="PCOS Risk Screening Report").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def build_report(lang, payload, result, by_input):
    """Return (pdf_bytes, language_used). Uses WeasyPrint so that Bangla text is shaped
    correctly; falls back to an English ReportLab page if that is not available."""
    try:
        from weasyprint import HTML
        html = report_html(TEXT[lang], lang, payload, result, by_input)
        return HTML(string=html, base_url=str(APP_DIR)).write_pdf(), lang
    except Exception:
        return build_report_basic(payload, result, by_input), "en"


# --------------------------------------------------------------- CSV upload

CSV_ROW_LIMIT = 5000

CSV_TEMPLATE = ("Age,Weight,Height,Testosterone,Antral follicle count,Menstrual cycle\n"
                "28,62,160,55,14,Regular\n"
                "24,70,158,95,28,Irregular\n")


def _key(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def match_columns(columns):
    """Map the file's column names to the fields used by the app. Returns
    (mapping, flags, ignored) where mapping is field -> original column."""
    mapping, ignored = {}, []
    irregularity_flag = False
    for column in columns:
        key = _key(column)
        field = None
        if key in ("age", "ageyears"):
            field = "age"
        elif key in ("weight", "weightkg", "bodyweight"):
            field = "weight"
        elif key in ("height", "heightcm"):
            field = "height"
        elif key == "bmi":
            field = "bmi"
        elif key.startswith("testosterone"):
            field = "testosterone"
        elif key in ("antralfolliclecount", "afc", "folliclecount", "antralfollicle"):
            field = "afc"
        elif key in ("menstrualcycle", "menstrualirregularity", "menstrual", "cycle"):
            field = "menstrual"
            irregularity_flag = "irregularity" in key
        if field and field not in mapping:
            mapping[field] = column
        else:
            ignored.append(column)
    return mapping, irregularity_flag, ignored


def parse_cycle(value):
    """Regular/Irregular (or 0/1, where 1 means irregular) -> app label or None."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    word = str(value).strip().lower()
    if word in ("regular", "normal", "0", "0.0"):
        return "Regular"
    if word in ("irregular", "1", "1.0"):
        return "Irregular"
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) or np.isinf(number) else None


def predict_table(frame, lang, model):
    """Validate every row of an uploaded table and predict the valid ones.
    Returns (result_frame, missing_columns, ignored_columns)."""
    text = TEXT[lang]
    mapping, _, ignored = match_columns(frame.columns)
    missing = [key for key in ("age", "testosterone", "afc", "menstrual") if key not in mapping]
    if "bmi" not in mapping and not ("weight" in mapping and "height" in mapping):
        missing += [key for key in ("weight", "height") if key not in mapping]
    if missing:
        return None, [text["need_columns"][key] for key in missing], ignored
    use_bmi = not ("weight" in mapping and "height" in mapping)

    records, valid_rows, feature_rows = [], [], []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        age = _number(row[mapping["age"]])
        testosterone = _number(row[mapping["testosterone"]])
        follicles = _number(row[mapping["afc"]])
        weight = None if use_bmi else _number(row[mapping["weight"]])
        height = None if use_bmi else _number(row[mapping["height"]])
        bmi_given = _number(row[mapping["bmi"]]) if use_bmi else None
        cycle_raw = row[mapping["menstrual"]]
        cycle = parse_cycle(cycle_raw)
        if use_bmi and bmi_given is None:
            bmi_given = float("nan")
        errors, warnings, bmi = validate_inputs(
            age, weight, height, testosterone, follicles, cycle, lang,
            bmi_value=bmi_given if use_bmi else None)
        if cycle is None and cycle_raw is not None and str(cycle_raw).strip() not in ("", "nan"):
            errors = [text["cycle_bad"]] + [e for e in errors if e != text["required"].format(
                field=text["fields"]["Menstrual cycle"])]
        records.append({"row": position, "age": age, "bmi": bmi, "testosterone": testosterone,
                        "follicles": follicles, "cycle": cycle, "errors": errors,
                        "warnings": warnings, "probability": None})
        if not errors:
            feature_rows.append([build_features(age, bmi, testosterone, follicles,
                                                MENSTRUAL_CHOICES[cycle])[name]
                                 for name in HYBRID_FEATURES])
            valid_rows.append(len(records) - 1)
    if feature_rows:
        probabilities = model.predict_proba(
            pd.DataFrame(feature_rows, columns=HYBRID_FEATURES))[:, 1]
        for index, probability in zip(valid_rows, probabilities):
            records[index]["probability"] = float(probability)

    shown = []
    for record in records:
        probability = record["probability"]
        band = risk_band(probability) if probability is not None else None
        if record["errors"]:
            note = " ".join(record["errors"])
        elif record["warnings"]:
            note = text["train_all"].format(items="; ".join(record["warnings"]))
        else:
            note = ""
        shown.append({
            text["col_row"]: record["row"],
            text["fields"]["Age"]: record["age"],
            "BMI": None if record["bmi"] is None else round(record["bmi"], 1),
            text["fields"]["Testosterone"]: record["testosterone"],
            "AFC": record["follicles"],
            text["fields"]["Menstrual cycle"]: (text[record["cycle"]] if record["cycle"] else None),
            text["col_prob"]: None if probability is None else round(probability * 100.0, 1),
            text["col_band"]: text["invalid"] if band is None else text["band_short"][band - 1],
            text["col_note"]: note})
    return pd.DataFrame(shown), [], ignored


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
[data-testid="stMain"], section.main {{ scrollbar-width: auto; scrollbar-color: #6366f1 rgba(127,127,127,0.25); }}
[data-testid="stMain"]::-webkit-scrollbar, section.main::-webkit-scrollbar {{ width: 12px; height: 12px; }}
[data-testid="stMain"]::-webkit-scrollbar-track, section.main::-webkit-scrollbar-track {{ background: rgba(127,127,127,0.25); }}
[data-testid="stMain"]::-webkit-scrollbar-thumb, section.main::-webkit-scrollbar-thumb {{ background: #6366f1; border-radius: 8px; border: 2px solid transparent; background-clip: padding-box; }}
.stApp, [data-testid="stHeader"] {{ background: var(--bg); }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: var(--side); }}
.stApp :is(p, label, span, li, h1, h2, h3, h4, h5, small, div[data-testid="stMarkdownContainer"]):not(.pcos-hero *):not(.pcos-rcard *):not(.pcos-pill):not(.tag):not(button *) {{ color: var(--fg); }}
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
BAND_STYLE = {
    1: ("linear-gradient(120deg, #8fdc86, #d4efc2)", "#4aa43f"),
    2: ("#f7c566", "#c98a1a"),
    3: ("linear-gradient(120deg, #c42a2a, #a81f1f)", "#7f1515"),
}


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
    for number, (low, high, _middle) in enumerate(BANDS, start=1):
        background, border = BAND_STYLE[number]
        rows += (f'<div class="pcos-gr"><div class="pcos-gl"><span class="pcos-swatch" '
                 f'style="background:{background};border:1px solid {border}"></span>'
                 f'<b>{low}–{high}%</b></div>'
                 f'<div class="pcos-gt">{text["band" + str(number)]}</div></div>')
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


def compute_result(model, payload, bmi, warnings):
    probability, agreement = predict(model, payload["age"], bmi, payload["testosterone"],
                                     payload["follicle_count"], payload["menstrual"])
    positive, uncertain = describe(probability)
    return {"probability": probability, "percent": probability * 100.0, "agreement": agreement,
            "n_trees": len(model.estimators_), "bmi": bmi, "positive": positive,
            "uncertain": uncertain, "band": risk_band(probability), "warnings": warnings}


def show_result(text, lang, payload, result, by_input):
    percent, band = result["percent"], result["band"]

    st.info(f"{text['bmi']}: {result['bmi']:.1f} kg/m²")
    notes = []
    if result["warnings"]:
        notes.append(text["train_all"].format(items="; ".join(result["warnings"])))
    if result["uncertain"]:
        notes.append(text["uncertain"])
    if notes:
        st.warning("\n\n".join(notes))

    icon = {1: "✅", 2: "⚠️", 3: "🚨"}[band]
    background, border = BAND_STYLE[band]
    marker = min(max(percent, 1.0), 99.0)
    st.markdown(
        f'<div class="pcos-rcard b{band}" style="background:{background};border:1px solid {border}">'
        f'<div class="v">{icon} {text["band" + str(band)]}</div>'
        f'<div class="e">{text["estimate"]}</div><div class="p">{percent:.1f}%</div>'
        f'<div class="pcos-gauge"><div class="pcos-tick" style="left:'
        f'{DECISION_THRESHOLD * 100:.0f}%"></div>'
        f'<div class="pcos-marker" style="left:{marker:.1f}%"></div></div>'
        f'<div class="n">{text["threshold_note"]}</div></div>',
        unsafe_allow_html=True)
    st.markdown(recommendation_html(text, band, xai_html(text, payload, result, by_input), lang),
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_report(lang, age, weight, height, testosterone, follicle_count, menstrual):
    """Build the PDF once per set of inputs and language."""
    payload = dict(age=age, weight=weight, height=height, testosterone=testosterone,
                   follicle_count=follicle_count, menstrual=menstrual)
    errors, warnings, bmi = validate_inputs(age, weight, height, testosterone, follicle_count,
                                            menstrual, lang)
    result = compute_result(load_model(), payload, bmi, warnings)
    explanation = explain(age, bmi, testosterone, follicle_count, menstrual)
    by_input = aggregate_by_input(explanation) if explanation else None
    return build_report(lang, payload, result, by_input)


def show_report_tab(text, lang, payload):
    if payload is None:
        st.markdown(f'<div class="pcos-box"><p>{text["report_wait"]}</p></div>',
                    unsafe_allow_html=True)
        return
    st.markdown(f'<div class="pcos-box accent"><h4>📄 {text["report_title"]}</h4>'
                f'<p>{text["report_intro"]}</p></div>', unsafe_allow_html=True)
    try:
        pdf, used = cached_report(lang, payload["age"], payload["weight"], payload["height"],
                                  payload["testosterone"], payload["follicle_count"],
                                  payload["menstrual"])
    except Exception:
        st.markdown(f'<div class="pcos-box warn"><p>{text["why_unavailable"]}</p></div>',
                    unsafe_allow_html=True)
        return
    if used != lang:
        st.caption(text["report_fallback"])
    st.download_button(text["report_btn"], data=pdf, file_name="PCOS_screening_report.pdf",
                       mime="application/pdf", use_container_width=True)


def show_csv_tab(text, lang, model):
    st.markdown(
        f'<div class="pcos-box accent"><h4>📄 {text["csv_title"]}</h4>'
        f'<p>{text["csv_help"]}</p></div>', unsafe_allow_html=True)
    st.download_button(text["csv_template"], data=CSV_TEMPLATE.encode("utf-8"),
                       file_name="PCOS_input_template.csv", mime="text/csv")
    uploaded = st.file_uploader(text["csv_upload"], type=["csv"], key="csv_file")
    if uploaded is None:
        st.session_state.pop("csv_result", None)
        return
    try:
        raw = uploaded.getvalue()
        try:
            frame = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False,
                                encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False, encoding="latin-1")
    except Exception:
        st.error(text["csv_unreadable"])
        return
    if frame.empty:
        st.error(text["csv_empty"])
        return
    if len(frame) > CSV_ROW_LIMIT:
        st.error(text["csv_toobig"].format(n=len(frame), limit=CSV_ROW_LIMIT))
        return
    file_id = (uploaded.name, len(raw), lang)
    if st.button(text["csv_predict"], use_container_width=True, key="csv_go"):
        table, missing, ignored = predict_table(frame, lang, model)
        st.session_state["csv_result"] = (file_id, table, missing, ignored)
    saved = st.session_state.get("csv_result")
    if not saved or saved[0] != file_id:
        return
    _, table, missing, ignored = saved
    if missing:
        st.error(text["csv_missing"].format(cols=", ".join(missing)))
        return
    bad = int((table[text["col_band"]] == text["invalid"]).sum())
    st.success(text["csv_summary"].format(ok=len(table) - bad, bad=bad))
    if ignored:
        st.caption(text["csv_ignored"].format(cols=", ".join(map(str, ignored))))
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(text["csv_download"], data=table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="PCOS_predictions.csv", mime="text/csv")


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

    tab_predict, tab_csv, tab_report = st.tabs(
        [text["tab_pred"], text["tab_csv"], text["tab_report"]])
    report_payload = None

    with tab_predict:
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
                boxes = "".join(
                    f'<div class="pcos-error"><span class="pcos-error-icon">!</span>'
                    f'<span>{message}</span></div>' for message in errors)
                st.markdown(f'<div class="pcos-error-list">{boxes}</div>', unsafe_allow_html=True)
            else:
                result = compute_result(model, payload, bmi, warnings)
                explanation = explain(payload["age"], bmi, payload["testosterone"],
                                      payload["follicle_count"], payload["menstrual"])
                by_input = aggregate_by_input(explanation) if explanation else None
                show_result(text, lang, payload, result, by_input)
                report_payload = payload

        st.markdown(
            f'<div class="pcos-box warn" style="margin-top:8px"><p>⚠️ '
            f'{text["disclaimer_body"]}</p></div>', unsafe_allow_html=True)

    with tab_csv:
        show_csv_tab(text, lang, model)

    with tab_report:
        show_report_tab(text, lang, report_payload)


if __name__ == "__main__":
    main()
