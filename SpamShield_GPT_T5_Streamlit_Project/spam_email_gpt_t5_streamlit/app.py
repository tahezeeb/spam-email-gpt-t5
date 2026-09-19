
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"

st.set_page_config(
    page_title="SpamShield GPT/T5",
    page_icon="📧",
    layout="wide"
)

st.title("📧 SpamShield — Generative AI for Email Spam Classification")
st.caption("AI/ML Final Project • NLP • GPT/T5 text generation • Streamlit deployment")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["text"] = data["text"].astype(str)
    return data

@st.cache_resource
def train_baseline(data_hash):
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, pred, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, pred, pos_label="spam", zero_division=0),
        "cm": confusion_matrix(y_test, pred, labels=["ham", "spam"]),
        "report": classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0)
    }
    return model, X_test, y_test, pred, proba, classes, metrics

@st.cache_resource
def load_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_with_t5(prompt, n=3, max_new_tokens=80):
    tokenizer, model = load_t5()
    outputs = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.9,
            num_return_sequences=1
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if text:
            outputs.append(text)
    return outputs

data = load_data()
model, X_test, y_test, pred, proba, classes, metrics = train_baseline(
    len(data),  # cache key
)

with st.sidebar:
    st.header("Project Controls")
    st.write(f"Dataset rows: **{len(data):,}**")
    st.write(f"Spam: **{(data.label == 'spam').sum():,}**")
    st.write(f"Ham: **{(data.label == 'ham').sum():,}**")
    st.divider()
    st.info(
        "The baseline classifier uses TF-IDF + Logistic Regression. "
        "The generative component uses Google's FLAN-T5-small to create synthetic email text."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Classify Email", "✨ Generate Synthetic Emails",
    "📊 Evaluation", "📘 Project Details"
])

with tab1:
    st.subheader("Classify an Email as Spam or Ham")
    sample = st.selectbox(
        "Try a sample",
        ["Custom", "Spam example", "Normal example"]
    )
    if sample == "Spam example":
        default_text = "URGENT! You have won a $5000 reward. Click the link now to verify your account and claim your prize."
    elif sample == "Normal example":
        default_text = "Hi team, the project meeting is scheduled for 10 AM tomorrow. Please review the agenda before joining."
    else:
        default_text = ""

    email = st.text_area(
        "Paste the email text",
        value=default_text,
        height=180,
        placeholder="Enter an email message here..."
    )

    if st.button("🚀 Classify Email", type="primary"):
        if not email.strip():
            st.warning("Please enter an email first.")
        else:
            pred_label = model.predict([email])[0]
            probs = model.predict_proba([email])[0]
            prob_map = dict(zip(classes, probs))
            if pred_label == "spam":
                st.error(f"🚨 SPAM — confidence: {prob_map['spam']:.1%}")
            else:
                st.success(f"✅ HAM / LEGITIMATE — confidence: {prob_map['ham']:.1%}")

            c1, c2 = st.columns(2)
            c1.metric("Spam probability", f"{prob_map['spam']:.1%}")
            c2.metric("Ham probability", f"{prob_map['ham']:.1%}")

            st.progress(float(prob_map["spam"]), text="Spam probability")

with tab2:
    st.subheader("Generate Synthetic Training Emails with FLAN-T5")
    col1, col2 = st.columns(2)
    with col1:
        label = st.selectbox("Email type", ["spam", "ham"])
        count = st.slider("Number of emails", 1, 8, 3)
    with col2:
        topic = st.text_input(
            "Topic / scenario",
            value="online shopping promotion" if label == "spam" else "work meeting reminder"
        )

    prompt = (
        f"Write a realistic {label} email about {topic}. "
        f"Return only the email body. Do not include explanations. "
        f"For spam, use realistic promotional or phishing-like language; "
        f"for ham, make it a normal legitimate email."
    )
    st.code(prompt, language="text")

    if st.button("✨ Generate Synthetic Emails"):
        with st.spinner("Loading FLAN-T5 and generating text..."):
            try:
                generated = generate_with_t5(prompt, n=count)
                if not generated:
                    st.warning("The model returned no text. Try again.")
                else:
                    for i, text in enumerate(generated, 1):
                        st.markdown(f"**Generated Email {i}**")
                        st.text_area(f"Email {i}", text, height=120, key=f"gen_{i}")
            except Exception as e:
                st.error("Generation failed. Check the deployment logs.")
                st.exception(e)

    st.markdown("### How augmentation fits the project")
    st.write(
        "The generated emails can be added to the labeled training set, after human review, "
        "to increase the amount of training text. The classifier can then be retrained on "
        "real + synthetic examples and compared with the baseline."
    )

with tab3:
    st.subheader("Baseline Model Evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1-score", f"{metrics['f1']:.3f}")

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots()
    ax.imshow(metrics["cm"])
    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, metrics["cm"][i, j], ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Classification Report")
    st.code(metrics["report"])

    st.markdown("#### Dataset Preview")
    st.dataframe(data.sample(min(10, len(data)), random_state=42), use_container_width=True)

with tab4:
    st.subheader("Project Overview")
    st.markdown("""
**Project title:** Generate Email Text for Spam Classification using NLP and GPT/T5

**Problem:** Spam filtering requires enough diverse labeled email text. Generative AI can create
additional synthetic examples, which can be reviewed and used as augmentation data.

**Architecture**
1. Starter labeled email dataset
2. Text preprocessing with TF-IDF
3. Baseline spam classifier: Logistic Regression
4. Generative AI: FLAN-T5-small
5. Synthetic email generation
6. Human review of generated text
7. Real + synthetic augmentation
8. Evaluation using Accuracy, Precision, Recall and F1-score
9. Streamlit web deployment

**Important:** Synthetic data is not automatically trustworthy. Generated messages should be reviewed
for duplication, bias, label errors and unrealistic patterns before being used for model training.
""")
    st.success("Deployment target: Streamlit Community Cloud")
    st.info("For a stronger final submission, upload a larger real-world spam/ham dataset and report baseline vs augmented metrics.")
