from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Experiment 5: Outlier Detection and Treatment (IQR and Z-score methods)

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")
cols = ["age", "absences", "G1", "G2", "G3"]
print("Dataset Shape :", df.shape)
print(df[cols].describe())

# Box plots before treatment
plt.figure(figsize=(10, 4))
sns.boxplot(data=df[cols], color="steelblue")
plt.title("Box Plots Before Outlier Treatment")
plt.show()

# ---------- IQR Method ----------
print("\n----- IQR Method -----")
iqr_outliers = {}
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    iqr_outliers[col] = (lower, upper)
    print(f"{col}: Q1={Q1}, Q3={Q3}, IQR={IQR}, Lower={lower}, Upper={upper}, Outliers={mask.sum()}")

# ---------- Z-score Method ----------
print("\n----- Z-score Method (|z| > 3) -----")
z_scores = np.abs(stats.zscore(df[cols]))
z_df = pd.DataFrame(z_scores, columns=cols)
for col in cols:
    print(f"{col}: Outliers = {(z_df[col] > 3).sum()}")

# ---------- Treatment ----------
# 1. Removal using Z-score
df_removed = df[(z_df < 3).all(axis=1)]
print("\nShape before removal :", df.shape)
print("Shape after removing Z-score outliers :", df_removed.shape)

# 2. Capping (clipping) using IQR limits
df_capped = df.copy()
for col in cols:
    lower, upper = iqr_outliers[col]
    df_capped[col] = df_capped[col].clip(lower, upper)

print("\nOutliers remaining after IQR capping:")
for col in cols:
    lower, upper = iqr_outliers[col]
    print(col, ((df_capped[col] < lower) | (df_capped[col] > upper)).sum())

# 3. Median replacement for 'absences'
df_median = df.copy()
lower, upper = iqr_outliers["absences"]
median_abs = df["absences"].median()
df_median.loc[(df["absences"] < lower) | (df["absences"] > upper), "absences"] = median_abs
print("\nAbsences max before:", df["absences"].max(), " after median replacement:", df_median["absences"].max())

# Box plots after treatment
plt.figure(figsize=(10, 4))
sns.boxplot(data=df_capped[cols], color="seagreen")
plt.title("Box Plots After IQR Capping")
plt.show()
