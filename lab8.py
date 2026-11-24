from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# Experiment 8: Logistic Regression for Binary Classification
# Target: pass (1) if final grade G3 >= 10, else fail (0)

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
df["pass"] = (df["G3"] >= 10).astype(int)
print("Class distribution:\n", df["pass"].value_counts())

features = ["G1", "studytime", "failures", "absences", "age", "Medu", "Fedu", "goout", "health"]
X = df[features]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel coefficients:")
print(pd.Series(model.coef_[0], index=features).round(4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}   (TP / (TP + FP))")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}   (TP / (TP + FN))")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}   (2*P*R / (P + R))")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Predicted probability for a few test students
probs = model.predict_proba(X_test)[:5, 1]
print("Predicted probability of passing (first 5 test students):", probs.round(3))
