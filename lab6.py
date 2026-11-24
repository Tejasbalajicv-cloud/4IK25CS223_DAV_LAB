from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Experiment 6: Feature Engineering - Normalization, Standardization and PCA

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
features = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health",
            "absences", "G1", "G2"]
X = df[features]
print("Original data:\n", X.describe().loc[["mean", "std", "min", "max"]])

# ---------- Normalization (Min-Max scaling to [0, 1]) ----------
minmax = MinMaxScaler()
X_norm = pd.DataFrame(minmax.fit_transform(X), columns=features)
print("\nAfter Normalization (Min-Max):\n", X_norm.describe().loc[["mean", "std", "min", "max"]].round(3))

# ---------- Standardization (mean 0, std 1) ----------
std = StandardScaler()
X_std = pd.DataFrame(std.fit_transform(X), columns=features)
print("\nAfter Standardization (Z-score):\n", X_std.describe().loc[["mean", "std", "min", "max"]].round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(X["absences"], kde=True, ax=axes[0], color="steelblue").set_title("Original absences")
sns.histplot(X_norm["absences"], kde=True, ax=axes[1], color="seagreen").set_title("Normalized absences")
sns.histplot(X_std["absences"], kde=True, ax=axes[2], color="darkorange").set_title("Standardized absences")
plt.tight_layout()
plt.show()

# ---------- PCA ----------
pca_full = PCA()
pca_full.fit(X_std)
explained = pca_full.explained_variance_ratio_
print("\nExplained variance ratio of each component:\n", explained.round(4))
print("Cumulative explained variance:\n", explained.cumsum().round(4))

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained) + 1), explained.cumsum(), marker="o")
plt.axhline(0.90, color="red", linestyle="--", label="90% variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Scree / Cumulative Variance Plot")
plt.legend()
plt.show()

n_90 = (explained.cumsum() >= 0.90).argmax() + 1
print(f"\nComponents needed to keep 90% variance: {n_90} (out of {len(features)})")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["G3"] = df["G3"]
print("\nReduced data (2 components):\n", pca_df.head())
print("\nPCA loadings:\n", pd.DataFrame(pca.components_, columns=features, index=["PC1", "PC2"]).round(3))

plt.figure(figsize=(6, 5))
sns.scatterplot(x="PC1", y="PC2", hue="G3", data=pca_df, palette="viridis")
plt.title("Data projected onto first two Principal Components")
plt.show()
