"""Train, compare, validate, and save Iris classification models."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison = {}
for name, candidate in models.items():
    scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="accuracy")
    comparison[name] = {"cv_mean_accuracy": float(scores.mean()), "cv_std": float(scores.std())}

best_name = max(comparison, key=lambda name: comparison[name]["cv_mean_accuracy"])
best_model = models[best_name]
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
test_accuracy = float(accuracy_score(y_test, pred))

joblib.dump(best_model, MODEL_DIR / "iris_model.joblib")
report = classification_report(y_test, pred, target_names=iris.target_names, output_dict=True)
results = {
    "selected_model": best_name,
    "test_accuracy": test_accuracy,
    "test_samples": int(len(y_test)),
    "cv_comparison": comparison,
    "classification_report": report,
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "class_names": iris.target_names.tolist(),
    "feature_names": iris.feature_names,
}
(ROOT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Model comparison (5-fold CV on training set):")
for name, scores in comparison.items():
    print(f"  {name}: {scores['cv_mean_accuracy']:.3f} ± {scores['cv_std']:.3f}")
print(f"\nSelected model: {best_name}")
print(f"Held-out test accuracy: {test_accuracy:.3f} ({len(y_test)} samples)")
print("Confusion matrix:")
print(results["confusion_matrix"])
print(f"Saved model: {MODEL_DIR / 'iris_model.joblib'}")
print(f"Saved metrics: {ROOT / 'metrics.json'}")
