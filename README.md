import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

# ---------- SETTINGS: edit these ----------
FILE = "your_data.csv"
DATE_COL = "order_date"                 # or None
TARGET = "sales"                        # number to analyse/predict
GROUPS = ["category", "region"]         # categorical columns to break down by
# ------------------------------------------

# 1. Load + clean
df = pd.read_csv(FILE, encoding="latin-1")
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)
print(df.shape)
print(df.info())

df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

for c in df.select_dtypes(include=np.number):
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(include="object"):
    df[c] = df[c].fillna(df[c].mode()[0])

# Cap outliers in the target for modelling
q1, q3 = df[TARGET].quantile([0.25, 0.75])
iqr = q3 - q1
df["target_capped"] = df[TARGET].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 2. Feature engineering
df["year"] = df[DATE_COL].dt.year
df["month"] = df[DATE_COL].dt.month
df["dayofweek"] = df[DATE_COL].dt.dayofweek

# 3. Analysis + visuals
monthly = df.set_index(DATE_COL)[TARGET].resample("M").sum()
monthly.plot(figsize=(11, 4), marker="o", title=f"Monthly total {TARGET}")
plt.show()
print("Peak month:", monthly.idxmax(), monthly.max())

df.groupby("month")[TARGET].mean().plot(kind="bar", title="Average by month (seasonality)")
plt.show()

for g in GROUPS:
    df.groupby(g)[TARGET].sum().sort_values().tail(10).plot(
        kind="barh", title=f"Total {TARGET} by {g}")
    plt.show()

num_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.show()

# 4. Prediction model
X = df.drop(columns=[TARGET, "target_capped", DATE_COL])
X = X[[c for c in X.columns if X[c].dtype != "object" or X[c].nunique() <= 30]]
X = pd.get_dummies(X, drop_first=True)
y = df["target_capped"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results[name] = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
print(pd.DataFrame(results).T.round(3))

# 5. Evaluation charts (Random Forest)
rf = models["Random Forest"]
pred = rf.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.3)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

pd.Series(rf.feature_importances_, index=X.columns).nlargest(10).sort_values().plot(
    kind="barh", title="Top 10 factors driving the prediction")
plt.show()
