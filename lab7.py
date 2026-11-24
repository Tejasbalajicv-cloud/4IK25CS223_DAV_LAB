from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Experiment 7: Simple and Multiple Linear Regression

df = pd.read_csv(Path(__file__).with_name("student_data.csv"), sep=";")

# ---------- Simple Linear Regression: G3 ~ G2 ----------
X = df[["G2"]]
y = df["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_test)
print("----- Simple Linear Regression (G3 ~ G2) -----")
print(f"Intercept: {slr.intercept_:.3f}")
print(f"Slope    : {slr.coef_[0]:.3f}")
print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color="steelblue", label="Actual")
plt.plot(X_test.sort_values("G2"), slr.predict(X_test.sort_values("G2")), color="red", label="Regression line")
plt.xlabel("G2 (Second Period Grade)")
plt.ylabel("G3 (Final Grade)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# ---------- Multiple Linear Regression ----------
features = ["G1", "G2", "studytime", "failures", "absences", "age", "Medu", "goout"]
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)
print("\n----- Multiple Linear Regression -----")
print("Intercept:", round(mlr.intercept_, 3))
print(pd.Series(mlr.coef_, index=features, name="Coefficient").round(4))
print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
n, p = X_test.shape
adj_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (n - 1) / (n - p - 1)
print(f"Adjusted R2: {adj_r2:.4f}")

# ---------- Checking Assumptions ----------
residuals = y_test - y_pred
print("\n----- Assumption Checks -----")
# 1. Linearity: actual vs predicted
# 2. Homoscedasticity: residuals vs fitted
# 3. Normality of residuals: histogram + Q-Q plot + Shapiro-Wilk
# 4. Multicollinearity: correlation among features / VIF
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].scatter(y_test, y_pred, color="steelblue")
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
axes[0, 0].set_title("Linearity: Actual vs Predicted")
axes[0, 0].set_xlabel("Actual G3")
axes[0, 0].set_ylabel("Predicted G3")
axes[0, 1].scatter(y_pred, residuals, color="darkorange")
axes[0, 1].axhline(0, color="red", linestyle="--")
axes[0, 1].set_title("Homoscedasticity: Residuals vs Fitted")
axes[0, 1].set_xlabel("Fitted values")
axes[0, 1].set_ylabel("Residuals")
sns.histplot(residuals, kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Normality: Residual Distribution")
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Normality: Q-Q Plot")
plt.tight_layout()
plt.show()

stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test on residuals: statistic={stat:.4f}, p-value={p_value:.4f}")
print("Residuals look normal" if p_value > 0.05 else "Residuals deviate from normality")
print(f"Mean of residuals: {residuals.mean():.4f}")

# Variance Inflation Factor (VIF) computed manually
print("\nVariance Inflation Factor (VIF):")
for col in features:
    others = [c for c in features if c != col]
    r2 = LinearRegression().fit(df[others], df[col]).score(df[others], df[col])
    print(f"  {col:10s}: {1 / (1 - r2):.2f}")

plt.figure(figsize=(7, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Multicollinearity Check - Feature Correlation")
plt.show()
