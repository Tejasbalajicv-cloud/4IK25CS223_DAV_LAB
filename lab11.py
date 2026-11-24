from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Experiment 11: Hierarchical Clustering - Dendrogram and Agglomerative Clustering

df = pd.read_csv(Path(__file__).with_name("student_performance_dataset.csv"))
features = ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores", "Final_Exam_Score"]
print(df[features].head())
X = StandardScaler().fit_transform(df[features])

# ---------- Dendrograms with different linkage methods ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, method in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
    Z = linkage(X, method=method)
    dendrogram(Z, ax=ax)
    ax.set_title(f"Dendrogram ({method} linkage)")
    ax.set_xlabel("Student index")
    ax.set_ylabel("Distance")
plt.tight_layout()
plt.show()

Z = linkage(X, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.axhline(y=6, color="red", linestyle="--", label="Cut at distance 6")
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Student index")
plt.ylabel("Euclidean distance")
plt.legend()
plt.show()

scipy_labels = fcluster(Z, t=6, criterion="distance")
print("Clusters obtained by cutting dendrogram at distance 6:", len(set(scipy_labels)))

# ---------- Agglomerative clustering ----------
for k in range(2, 6):
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    print(f"K={k}: silhouette = {silhouette_score(X, labels):.4f}")

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["Cluster"] = agg.fit_predict(X)
print("\nCluster sizes:\n", df["Cluster"].value_counts().sort_index())
print("\nCluster profiles:\n", df.groupby("Cluster")[features].mean().round(2))
print("\nPass/Fail by cluster:\n", pd.crosstab(df["Cluster"], df["Pass_Fail"]))

plt.figure(figsize=(6, 5))
sns.scatterplot(x="Study_Hours_per_Week", y="Final_Exam_Score", hue="Cluster",
                style="Pass_Fail", data=df, palette="Set1", s=80)
plt.title("Agglomerative Clustering (3 clusters)")
plt.show()
