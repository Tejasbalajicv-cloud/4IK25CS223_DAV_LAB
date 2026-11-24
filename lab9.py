from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

# Experiment 9: Decision Tree Classifier on a real-world dataset
# (UCI Student Performance dataset - predict pass/fail)

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
df["result"] = (df["G3"] >= 10).map({True: "Pass", False: "Fail"})

# Encode a few categorical columns as 0/1
df["sex"] = df["sex"].map({"F": 0, "M": 1})
df["internet"] = df["internet"].map({"no": 0, "yes": 1})
df["higher"] = df["higher"].map({"no": 0, "yes": 1})

features = ["G1", "G2", "studytime", "failures", "absences", "age", "sex",
            "Medu", "goout", "internet", "higher"]
X = df[features]
y = df["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ---------- Unpruned tree ----------
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)
print("Unpruned tree depth:", full_tree.get_depth(), " leaves:", full_tree.get_n_leaves())
print(f"Train accuracy: {full_tree.score(X_train, y_train):.4f}")
print(f"Test accuracy : {full_tree.score(X_test, y_test):.4f}")

# ---------- Effect of max_depth ----------
depths = range(1, 11)
train_acc, test_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(m.score(X_train, y_train))
    test_acc.append(m.score(X_test, y_test))
plt.figure(figsize=(6, 4))
plt.plot(depths, train_acc, marker="o", label="Train")
plt.plot(depths, test_acc, marker="o", label="Test")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Depth")
plt.legend()
plt.show()

# ---------- Final (pruned) tree ----------
tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("\n----- Pruned Tree (max_depth=3) -----")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Tree rules:\n", export_text(tree, feature_names=features))

importance = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)
print("Feature importance:\n", importance.round(4))

# ---------- Visualization ----------
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=features, class_names=tree.classes_, filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree (max_depth=3)")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x=importance.values, y=importance.index, color="steelblue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]), annot=True, fmt="d",
            cmap="Greens", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
