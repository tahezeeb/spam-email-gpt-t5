# SpamShield — GPT/T5 Email Generation + Spam Classification

## Project
**Generate email text for spam classification using NLP and GPT/T5**

This project follows the internship brief's Project 4 direction: **"Generate email text for spam classification" using NLP and GPT/T5**.

## Features
- Spam vs ham email classification
- TF-IDF + Logistic Regression baseline
- FLAN-T5-small synthetic email generation
- Synthetic data augmentation concept
- Accuracy, precision, recall and F1 evaluation
- Confusion matrix
- Streamlit web UI
- Ready for Streamlit Community Cloud deployment

## Folder structure
```text
spam_email_gpt_t5_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── emails.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_t5_generation.ipynb
│   └── 03_model_training_evaluation.ipynb
├── reports/
│   └── final_report.md
└── .streamlit/
    └── config.toml
```

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first time you use generation, Hugging Face Transformers downloads `google/flan-t5-small`.

## Expected demo flow
1. Open **Classify Email** and test spam/ham examples.
2. Open **Generate Synthetic Emails** and generate spam/ham samples.
3. Explain that generated emails can be reviewed and added to the training data.
4. Open **Evaluation** and show Accuracy / Precision / Recall / F1 and the confusion matrix.
5. Deploy the repository using Streamlit Community Cloud.

## Important academic note
The bundled dataset is a starter/demo dataset created for this project package. For the final academic experiment,
replace or supplement it with a real public spam/ham dataset, document its source, split the data before augmentation,
and report baseline vs augmented metrics on an untouched test set.

## Deployment
Streamlit Community Cloud can deploy a GitHub repository containing the app and `requirements.txt`.
