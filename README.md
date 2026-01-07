# AutoJudge: Predicting Programming Problem Difficulty

## Running the Project Locally

### 1. Download Required Files

Ensure the following files are present in the same directory:

- `AutoJudge.ipynb` (GitHub)
- `app.py` (GitHub)
- `problems_data.jsonl` (GitHub)
- `best_classification_model.pkl` (Link: "https://drive.google.com/drive/folders/11XE8AT5dUJQN8k91vmJcs3jKAKFmb4yv?usp=sharing")
- `best_regression_model.pkl` (Link: "https://drive.google.com/drive/folders/11XE8AT5dUJQN8k91vmJcs3jKAKFmb4yv?usp=sharing)

---

### 2. Install Dependencies

Install all required Python packages to run the app:

```bash
pip install streamlit scikit-learn pandas numpy joblib spacy
python -m spacy download en_core_web_sm
```


### 3. Run the Web App on "http://localhost:8501"

```bash
python -m streamlit run app.py
```
---

## Link to the Demo video: "https://drive.google.com/drive/folders/11XE8AT5dUJQN8k91vmJcs3jKAKFmb4yv?usp=sharing"

---

## Project Overview

Online competitive programming platforms such as Codeforces, CodeChef, and Kattis assign difficulty labels (Easy / Medium / Hard) and numerical difficulty scores to problems. These labels are typically determined through human judgment and community feedback.

**AutoJudge** is a machine learning–based system that automatically predicts:
- **Problem Difficulty Class** → Easy / Medium / Hard (Classification)
- **Problem Difficulty Score** → Numerical value (Regression)

The prediction is made **using only the textual content of the problem**, without any user statistics or historical submissions.

The project also includes a **simple web interface** that allows users to paste a problem statement and instantly view predicted difficulty.

---

## Dataset Used

The dataset (problems_data.jsonl) consists of programming problems collected from online judges.  
Each data sample includes the following fields:

- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical difficulty score)
- `sample_io` (not used for modeling)
- `url` (not used for modeling)

Only **textual fields** are used for feature extraction, as required by the project constraints.

---

## Approach

### 1. Text Combination
The following fields are combined into a single textual input:
- Problem description
- Input description
- Output description

This ensures the full problem context is captured.

---

### 2. Text Preprocessing

Text preprocessing is carefully designed to avoid feature leakage and double counting:

- Unicode normalization
- Removal of LaTeX expressions and subscripts
- Removal of punctuation and formatting artifacts
- Stopword removal
- Lemmatization using spaCy
- Lowercasing and whitespace normalization

---

### 3. Feature Engineering

To capture both **semantic meaning** and **structural complexity**, multiple feature types are used:

#### a) TF-IDF Features
- Applied on cleaned problem text
- Vocabulary size limited to 2000 features to reduce noise
- Rare and overly common words filtered

#### b) Keyword-Based Features
Weighted keyword counts for algorithmic concepts such as:
- `dp`, `graph`, `greedy`, `flow`, `binary search`, `dfs`, `bfs`, etc.

Keywords are:
- Counted and converted into numeric features
- **Removed from text before TF-IDF** to prevent double counting

#### c) Structural Features
- Number of mathematical symbols (`+ - * / < > = ≤ ≥ ≠`)
- Length of the cleaned text (number of tokens)

---

### 4. Unified Preprocessing Pipeline

All preprocessing and feature extraction steps are implemented inside a **single `sklearn Pipeline`** using a `ColumnTransformer`.

This ensures:
- Identical preprocessing during training and inference
- Safe deployment without manual feature mismatch
- Robust integration with the web interface

---

## Models Used

### Classification Models
The following models were evaluated (Hyperparameters were tuned) using cross-validation:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multinomial Naive Bayes

The **best-performing model** (based on accuracy) was selected and saved as "best_classification_model.pkl".

---

### Regression Models
The following models were evaluated using cross-validation:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The **best regression model** was selected based on MAE and RMSE and saved as "best_regression_model.pkl".

---

## 📊 Evaluation Metrics:

### Classification Metrics (Plots were plotted for all classification models)
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- ROC Curve (macro-averaged)
- Precision-Recall Curve (macro-averaged)
- Confusion Matrix

### Classification Results

The following table shows the performance of different classification models
on the test set. Accuracy is used as the primary evaluation metric.

| Model                     | Accuracy |
|---------------------------|----------|
| **Random Forest**         | **0.5371** |
| Multinomial Naive Bayes   | 0.5067 |
| Logistic Regression       | 0.5055 |
| Support Vector Machine    | 0.4969 |

**Best Classification Model:** Random Forest  

---

### Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Regression Results

The regression task predicts a numerical difficulty score for each problem.
Performance is evaluated using **Mean Absolute Error (MAE)** and
**Root Mean Squared Error (RMSE)**.

| Model                 | MAE ↓ | RMSE ↓ |
|----------------------|------:|-------:|
| **Random Forest**        | **1.7146** | **2.0364** |
| Ridge Regression     | 1.7118 | 2.0443 |
| Lasso Regression     | 1.6975 | 2.0472 |
| Gradient Boosting    | 1.7076 | 2.0476 |
| Linear Regression    | 2.3675 | 2.9649 |

**Best Regression Model:** Random Forest  

---

## Web Interface (Streamlit)

A lightweight **Streamlit-based web interface** is provided for interactive usage.

### Features
- Text boxes for:
  - Problem description
  - Input description
  - Output description
- One-click prediction
- Displays:
  - Predicted difficulty class
  - Predicted difficulty score

### Key Design Choice
The web interface **does not re-implement preprocessing**.  
Instead, it passes user input directly to the trained pipelines, ensuring full consistency with training.

---

## Author

**Name:** Architkumar J. Modi  
**Branch:** Mathematics and Computing  
**Enrollment Number:** 23323022

---
