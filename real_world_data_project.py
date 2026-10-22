"""
Real-world Data Project - end-to-end pipeline (default domain: RETAIL sales)
Usage: python real_world_data_project.py your_data.csv

Steps: load -> clean -> feature engineering -> EDA charts -> prediction model
       -> evaluation -> auto-written findings (project_report.md)

HOW TO ADAPT: edit the CONFIG block below to match your dataset's column names.
  Retail  (Superstore / Online Retail): date, category, sales/profit
  Finance (stock prices):               date, ticker, close price
  Health  (patient records):            admission date, department, cost/outcome
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================== CONFIG =======================================
CONFIG = {
    "date_col": "order_date",          # date column (or None)
    "target_col": "sales",             # number you want to analyse/predict
    "group_cols": ["category", "region"],   # categorical columns to break down by
}
# ============================================================================

sns.set_theme(style="whitegrid")
os.makedirs("project_charts", exist_ok=True)
findings = []

path = sys.argv[1] if len(sys.argv) > 1 else "your_data.csv"
df = pd.read_csv(path, encoding="latin-1")
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)

date_col = CONFIG["date_col"]
target = CONFIG["target_col"]
groups = [g for g in CONFIG["group_cols"] if g in df.columns]

if target not in df.columns:
    sys.exit(f"Target column '{target}' not found. Columns are: {list(df.columns)}\n"
             "Edit CONFIG at the top of the script.")

# 1. CLEANING -----------------------------------------------------------------
n0 = len(df)
df = df.drop_duplicates()
if date_col and date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
else:
    date_col = None

num_cols = df.select_dtypes(include=np.number).columns.tolist()
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(include="object"):
    df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "unknown")

# Cap extreme outliers in the target (IQR) for modelling, keep a raw copy for totals
q1, q3 = df[target].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((df[target] < low) | (df[target] > high)).sum())
df["target_capped"] = df[target].clip(low, high)

findings.append(f"- Cleaned data: {n0:,} rows -> {len(df):,} after removing duplicates/invalid dates.")
findings.append(f"- {n_out:,} outliers in `{target}` were capped for modelling.")
findings.append(f"- Total `{target}`: {df[target].sum():,.0f}; average per record: {df[target].mean():,.2f}.")

# 2. FEATURE ENGINEERING ------------------------------------------------------
if date_col:
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter

# 3. VISUALS ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df[target], bins=40, kde=True)
plt.title(f"Distribution of {target}")
plt.tight_layout()
plt.savefig("project_charts/01_target_distribution.png", dpi=150)
plt.close()

if date_col:
    monthly = df.set_index(date_col)[target].resample("M").sum()
    plt.figure(figsize=(11, 4))
    monthly.plot(marker="o")
    plt.title(f"Monthly total {target}")
    plt.tight_layout()
    plt.savefig("project_charts/02_monthly_trend.png", dpi=150)
    plt.close()
    best_month = monthly.idxmax()
    findings.append(f"- Peak month for `{target}`: {best_month:%B %Y} ({monthly.max():,.0f}).")

    by_month = df.groupby("month")[target].mean()
    plt.figure(figsize=(8, 4))
    by_month.plot(kind="bar")
    plt.title(f"Average {target} by calendar month (seasonality)")
    plt.tight_layout()
    plt.savefig("project_charts/03_seasonality.png", dpi=150)
    plt.close()
    findings.append(f"- Strongest calendar month on average: month {int(by_month.idxmax())}.")

for g in groups:
    agg = df.groupby(g)[target].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    agg.sort_values().plot(kind="barh")
    plt.title(f"Total {target} by {g}")
    plt.tight_layout()
    plt.savefig(f"project_charts/04_{g}_totals.png", dpi=150)
    plt.close()
    share = agg.iloc[0] / df[target].sum() * 100
    findings.append(f"- Top `{g}`: **{agg.index[0]}** contributes {share:.1f}% of total `{target}`.")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) > 2:
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=len(num_cols) <= 12, fmt=".2f",
                cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig("project_charts/05_correlation.png", dpi=150)
    plt.close()
    tc = df[num_cols].corr()[target].drop([target, "target_capped"], errors="ignore")
    if len(tc):
        top = tc.abs().idxmax()
        findings.append(f"- Numeric feature most correlated with `{target}`: **{top}** ({tc[top]:.2f}).")

# 4. PREDICTION MODEL ---------------------------------------------------------
drop_cols = [target, "target_capped"] + ([date_col] if date_col else [])
X = df.drop(columns=drop_cols)
# drop high-cardinality text (ids, names) that would explode the feature count
X = X[[c for c in X.columns if X[c].dtype != "object" or X[c].nunique() <= 30]]
X = pd.get_dummies(X, drop_first=True)
y = df["target_capped"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
}
scores = {}
best_name, best_pred = None, None
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    r2 = r2_score(y_test, pred)
    scores[name] = {
        "R2": r2,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
    if best_name is None or r2 > scores[best_name]["R2"]:
        best_name, best_pred = name, pred

score_df = pd.DataFrame(scores).T.round(3)
print(score_df)
findings.append(f"- Best model: **{best_name}** (R2 = {scores[best_name]['R2']:.2f}, "
                f"MAE = {scores[best_name]['MAE']:.2f}).")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.3)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted - {best_name}")
plt.tight_layout()
plt.savefig("project_charts/06_actual_vs_predicted.png", dpi=150)
plt.close()

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(8, 5))
imp.sort_values().plot(kind="barh", title="Top 10 factors driving the prediction")
plt.tight_layout()
plt.savefig("project_charts/07_feature_importance.png", dpi=150)
plt.close()
findings.append(f"- Most important predictor: **{imp.index[0]}**.")

# 5. REPORT -------------------------------------------------------------------
report = [
    "# Real-world Data Project Report",
    f"\n**Dataset:** `{os.path.basename(path)}` | **Target:** `{target}`",
    "\n## Problem Statement\n\n(Write 2-3 lines: what business question are you answering?)",
    "\n## Key Findings (auto-generated - refine in your own words)\n",
    "\n".join(findings),
    "\n## Model Performance\n",
    score_df.to_markdown(),
    "\n## Conclusions and Recommendations\n\n1. ...\n2. ...\n3. ...",
    "\n## Limitations\n\n(e.g. limited time range, missing features, outliers capped)",
]
with open("project_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nDone. Report: project_report.md | Charts: project_charts/")
