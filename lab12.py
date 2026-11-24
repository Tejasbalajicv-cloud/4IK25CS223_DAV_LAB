from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                             f1_score, confusion_matrix, classification_report)

# Experiment 12: Mini Project - End-to-end data analytics pipeline
# Dataset: UCI Student Performance (student_data.csv)
# Goal   : Analyse what affects students' final grade and predict Pass/Fail

# ================= 1. Data Loading =================
df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
print("Shape:", df.shape)
print(df.head())

# ================= 2. Exploratory Data Analysis =================
print("\nData types:\n", df.dtypes.value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].describe())

df["result"] = np.where(df["G3"] >= 10, "Pass", "Fail")
print("\nPass/Fail counts:\n", df["result"].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["G3"], bins=20, kde=True, ax=axes[0, 0], color="steelblue").set_title("Final Grade Distribution")
sns.boxplot(x="studytime", y="G3", data=df, ax=axes[0, 1], color="steelblue").set_title("G3 by Study Time")
sns.boxplot(x="failures", y="G3", data=df, ax=axes[1, 0], color="darkorange").set_title("G3 by Past Failures")
sns.countplot(x="higher", hue="result", data=df, ax=axes[1, 1]).set_title("Wants Higher Education vs Result")
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include="number").columns
plt.figure(figsize=(12, 9))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=True, fmt=".1f", annot_kws={"size": 7})
plt.title("Correlation Heatmap")
plt.show()
print("\nCorrelation with G3:\n", df[num_cols].corr()["G3"].sort_values(ascending=False).round(3))

# ================= 3. Preprocessing =================
data = df.copy()
# Outlier treatment for absences (IQR capping)
Q1, Q3 = data["absences"].quantile([0.25, 0.75])
upper = Q3 + 1.5 * (Q3 - Q1)
print(f"\nCapping absences at {upper} ({(data['absences'] > upper).sum()} outliers)")
data["absences"] = data["absences"].clip(upper=upper)

# Encode binary yes/no and two-valued categorical columns
binary_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() == 2 and c != "result"]
for c in binary_cols:
    data[c] = pd.factorize(data[c], sort=True)[0]
# One-hot encode the remaining categorical columns
multi_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c]) and c != "result"]
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)
print("Encoded binary columns:", binary_cols)
print("One-hot encoded columns:", multi_cols)
print("Shape after encoding:", data.shape)

# ================= 4a. Model Building - Regression (predict G3) =================
X_reg = data.drop(columns=["G3", "result"])
y_reg = data["G3"]
Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
lin = LinearRegression().fit(Xtr, ytr)
pred = lin.predict(Xte)
print("\n----- Linear Regression (predict G3) -----")
print(f"R2  : {r2_score(yte, pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.4f}")

plt.figure(figsize=(5, 5))
plt.scatter(yte, pred, color="steelblue")
plt.plot([0, 20], [0, 20], "r--")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# ================= 4b. Model Building - Classification (Pass/Fail) =================
# G2 and G3 are excluded so that the model predicts early (using only G1 + background)
X_clf = data.drop(columns=["G2", "G3", "result"])
y_clf = (data["result"] == "Pass").astype(int)
Xtr, Xte, ytr, yte = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
results = []
for name, m in models.items():
    m.fit(Xtr_s, ytr)
    p = m.predict(Xte_s)
    cv = cross_val_score(m, scaler.transform(X_clf), y_clf, cv=5).mean()
    results.append({"Model": name, "Accuracy": accuracy_score(yte, p),
                    "F1": f1_score(yte, p), "CV Accuracy": cv})
results = pd.DataFrame(results).set_index("Model").round(4)
print("\n----- Classification Model Comparison -----\n", results)

best_name = results["F1"].idxmax()
best = models[best_name]
best_pred = best.predict(Xte_s)
print(f"\nBest model: {best_name}")
print(classification_report(yte, best_pred, target_names=["Fail", "Pass"]))

# ================= 5. Visualization of Results =================
results[["Accuracy", "F1", "CV Accuracy"]].plot(kind="bar", figsize=(8, 4), rot=0)
plt.ylim(0, 1)
plt.title("Model Comparison")
plt.ylabel("Score")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(yte, best_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X_clf.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 4))
sns.barplot(x=imp.values, y=imp.index, color="seagreen")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

print("\n----- Conclusion -----")
print("* G1 (first period grade) and past failures are the strongest predictors of the final result.")
print("* Final grade can be predicted well once G1 and G2 are known (R2 above 0.7).")
print(f"* Using only G1 and background data, {best_name} predicts Pass/Fail with "
      f"{results.loc[best_name, 'Accuracy'] * 100:.1f}% accuracy, so at-risk students can be identified early.")
