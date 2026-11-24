from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Experiment 10: K-Means Clustering with Elbow Method

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
features = ["studytime", "absences", "G1", "G2", "G3", "goout", "failures"]
X = StandardScaler().fit_transform(df[features])

# ---------- Elbow method ----------
k_values = range(1, 11)
wcss = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
    print(f"K={k}: WCSS={km.inertia_:.2f}")

plt.figure(figsize=(6, 4))
plt.plot(k_values, wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()

print("\nSilhouette scores:")
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"K={k}: {silhouette_score(X, labels):.4f}")

# ---------- Final model ----------
optimal_k = 3  # chosen from the elbow curve
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)
print(f"\nCluster sizes (K={optimal_k}):\n", df["Cluster"].value_counts().sort_index())
print("\nCluster profiles (mean values):\n", df.groupby("Cluster")[features].mean().round(2))

# ---------- Visualizing clusters ----------
plt.figure(figsize=(6, 5))
sns.scatterplot(x="G1", y="G3", hue="Cluster", data=df, palette="Set1")
plt.title("K-Means Clusters: G1 vs G3")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["Cluster"], palette="Set1")
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c="black", marker="X", label="Centroids")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (PCA 2D projection)")
plt.legend()
plt.show()
