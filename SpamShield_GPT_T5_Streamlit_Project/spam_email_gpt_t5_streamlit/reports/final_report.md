# Final Project Report
## Generate Email Text for Spam Classification using NLP and GPT/T5

### 1. Abstract
This project investigates the use of Generative AI for synthetic email text generation and spam classification.
A TF-IDF + Logistic Regression classifier is used as the baseline, while FLAN-T5-small is used to generate
additional synthetic email text. The goal is to study whether reviewed synthetic examples can augment the
training corpus and support spam classification.

### 2. Problem Statement
Email spam filtering needs diverse and representative labeled examples. Synthetic text generation can provide
additional training examples when real labeled data is limited.

### 3. Objectives
- Build an NLP spam classifier.
- Generate synthetic spam and legitimate email text using a T5-family generative model.
- Demonstrate a real + synthetic augmentation workflow.
- Evaluate the classifier with accuracy, precision, recall and F1-score.
- Deploy the application using Streamlit.

### 4. Technology Stack
- Python
- Pandas / NumPy
- Scikit-learn
- Hugging Face Transformers
- FLAN-T5-small
- PyTorch
- Streamlit
- Matplotlib

### 5. Methodology
1. Load labeled email text.
2. Split into training and test data.
3. Convert text into TF-IDF features.
4. Train Logistic Regression.
5. Generate synthetic email text with FLAN-T5-small.
6. Human-review generated samples.
7. Add approved synthetic text to the training set.
8. Retrain the classifier.
9. Evaluate both baseline and augmented models on the same untouched test set.

### 6. Evaluation
Record the following for the baseline and augmented experiments:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline | TBD | TBD | TBD | TBD |
| Real + Synthetic | TBD | TBD | TBD | TBD |

### 7. Results and Discussion
Do not claim improvement unless it is measured on an untouched test set. Discuss false positives,
false negatives, label quality, synthetic-data diversity and possible data leakage.

### 8. Limitations
- Synthetic text can contain artifacts or incorrect labels.
- A small demo dataset is not representative of all real email traffic.
- A public spam dataset may contain language and formatting biases.
- Generative model downloads increase deployment size and startup time.

### 9. Conclusion
The application demonstrates an end-to-end Generative AI + NLP workflow for spam classification and provides
a deployable interface for classification and synthetic email generation.

### 10. Future Work
- Use a larger public dataset.
- Compare GPT-style and T5-style generation.
- Add semantic-duplicate filtering.
- Compare Logistic Regression, SVM and transformer classifiers.
- Track baseline vs augmented performance across multiple random seeds.
