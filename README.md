# PetalWise — Modern Iris Classifier

A modern Streamlit ML dashboard for classifying Iris flowers. It compares Logistic Regression, Random Forest, and Support Vector Machine using 5-fold stratified cross-validation on the training set, selects the best mean CV performer, then evaluates it once on a held-out test set.

## Features
- Interactive measurements and predictions
- Original local SVG flower illustrations
- Candidate model comparison using cross-validation
- Held-out test accuracy, confusion matrix, and per-class precision/recall/F1
- Saved model and reproducible metrics

## Technology
Python, Streamlit, scikit-learn, pandas, NumPy, Joblib.

## Run on Windows
Open PowerShell in this folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If PowerShell blocks activation, these commands intentionally use the virtual environment's Python directly, so activation is not required. If the environment does not exist, create it first with `python -m venv .venv`.

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Deployment URLs
- GitHub repository: `ADD_YOUR_GITHUB_URL`
- AWS deployment: `ADD_YOUR_AWS_URL`
- GCP deployment: `ADD_YOUR_GCP_URL`
- Optional Streamlit Community Cloud demo: `ADD_YOUR_STREAMLIT_URL`

Replace these placeholders only with URLs that you have actually deployed and tested.

## Accuracy note
The Iris dataset is small and relatively easy to classify. A higher test score on one split does not prove a model is universally better. The script compares models with cross-validation on training data and uses the held-out test set for final evaluation. Do not edit metrics to inflate the score.

## University submission
This starter is not a substitute for the official MLOps practical manual. Complete every assigned practical and add genuine screenshots of your own implementation and results.
