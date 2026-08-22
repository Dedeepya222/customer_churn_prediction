import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/Telco_Churn.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# ============================================================
# CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()

# Remove unwanted columns
drop_columns = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Fill missing values
df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)


# ============================================================
# ENCODE CATEGORICAL COLUMNS
# ============================================================

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col].astype(str))


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(
    kind="bar"
)

plt.title("Top 10 Features Influencing Customer Churn")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully as churn_model.pkl")