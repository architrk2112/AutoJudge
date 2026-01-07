import streamlit as st
import joblib
import pandas as pd
import re
import spacy

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("best_classification_model.pkl")
    reg = joblib.load("best_regression_model.pkl")
    return clf, reg

clf_model, reg_model = load_models()

# -----------------------------
# NLP setup
# -----------------------------
nlp = spacy.load("en_core_web_sm")

MATH_SYMBOL_PATTERN = r"[+\-*/%^=<>≤≥≠]"

keyword_weights = {
    "dp": 2.5,
    "graph": 2.0,
    "greedy": 1.8,
    "math": 1.5,
    "geometry": 1.8,
    "string": 1.2,
    "shortest path": 3.0,
    "flow": 3.0,
    "matching": 3.0,
    "number theory": 2.5,
    "binary search": 2.0,
    "recursion": 1.5,
    "tree": 1.5,
    "dfs": 1.5,
    "bfs": 1.5,
    "combinatoric": 2.0,
    "modulo": 1.3,
    "xor": 1.3,
    "query": 1.2
}

# -----------------------------
# Preprocessing
# -----------------------------
def normalize_text(text):
    """Normalize escapes and unicode, but KEEP math symbols"""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"_\w+", "", text)
    text = re.sub(r":\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def preprocess_text_no_math_removal(text):
    """
    Clean text WITHOUT removing math symbols.
    Used ONLY before counting math symbols.
    """
    text = normalize_text(text)
    return text

def preprocess_text_remove_math(text):
    """
    Full cleaning AFTER math symbol counting.
    """
    # remove math & comparison symbols
    text = re.sub(r"[<>^_=]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(tokens)

def count_math_symbols(text):
    return len(re.findall(MATH_SYMBOL_PATTERN, text))

def extract_features(raw_text):
    # 1️⃣ Normalize text but KEEP math symbols
    normalized = preprocess_text_no_math_removal(raw_text)

    # 2️⃣ Count math symbols
    math_count = count_math_symbols(normalized)

    # 3️⃣ Remove math symbols + lemmatize
    clean = preprocess_text_remove_math(normalized)

    # 4️⃣ Keyword extraction
    sorted_keywords = sorted(keyword_weights.keys(), key=len, reverse=True)
    combined_pattern = r"\b(" + "|".join(map(re.escape, sorted_keywords)) + r")\b"

    features = {}

    for kw, weight in keyword_weights.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        features[kw] = len(re.findall(pattern, clean)) * weight

    # 5️⃣ Strip keywords from clean_text
    clean = re.sub(combined_pattern, " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # 6️⃣ Remaining numeric features
    features["clean_text"] = clean
    features["math_symbol_count"] = math_count
    features["text_len"] = len(clean.split())

    return features

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.caption("Programming Problem Difficulty Predictor")

description = st.text_area("Problem Description", height=200)
input_desc = st.text_area("Input Description", height=120)
output_desc = st.text_area("Output Description", height=120)

if st.button("🔍 Predict"):
    if not description.strip():
        st.warning("Problem description is required.")
    else:
        combined_text = f"{description} {input_desc} {output_desc}"

        feature_dict = extract_features(combined_text)
        input_df = pd.DataFrame([feature_dict])

        pred_class = clf_model.predict(input_df)[0]
        pred_score = reg_model.predict(input_df)[0]

        label_map = {0: "Easy", 1: "Medium", 2: "Hard"}

        st.success("### Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Difficulty Class", label_map[pred_class])

        with col2:
            st.metric("Difficulty Score", round(float(pred_score), 2))