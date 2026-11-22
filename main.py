
# ============================================================
# PASSENGER SATISFACTION PREDICTION - MACHINE LEARNING PROJECT
# ============================================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. Load Dataset
# ============================================================

if Path("dataset/train.csv").exists():
    train_path = "dataset/train.csv"
    test_path = "dataset/test.csv"
else:
    train_path = "train.csv"
    test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train Dataset Shape:", train.shape)
print("Test Dataset Shape:", test.shape)


# ============================================================
# 3. Initial Dataset Analysis
# ============================================================

print("\nFirst 5 Rows:")
print(train.head())

print("\nDataset Information:")
train.info()

print("\nMissing Values:")
print(train.isnull().sum())

print("\nDuplicate Rows:", train.duplicated().sum())

print("\nPassenger Satisfaction:")
print(train["satisfaction"].value_counts())

print("\nStatistical Summary:")
print(train.describe())


# ============================================================
# 4. Exploratory Data Analysis (EDA)
# ============================================================

sns.set_theme(style="whitegrid")

# 1. Satisfaction Distribution

plt.figure(figsize=(8, 5))
sns.countplot(data=train, x="satisfaction")
plt.title("Passenger Satisfaction Distribution")
plt.xlabel("Satisfaction")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# 2. Gender Distribution

plt.figure(figsize=(7, 5))
sns.countplot(data=train, x="Gender")
plt.title("Passenger Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# 3. Customer Type Distribution

plt.figure(figsize=(8, 5))
sns.countplot(data=train, x="Customer Type")
plt.title("Customer Type Distribution")
plt.xlabel("Customer Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# 4. Satisfaction by Travel Class

plt.figure(figsize=(8, 5))
sns.countplot(
    data=train,
    x="Class",
    hue="satisfaction"
)
plt.title("Satisfaction by Travel Class")
plt.xlabel("Travel Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()


# 5. Age Distribution

plt.figure(figsize=(8, 5))
sns.histplot(
    data=train,
    x="Age",
    bins=30,
    kde=True
)
plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 6. Correlation Heatmap

numeric_data = train.select_dtypes(include=np.number)

plt.figure(figsize=(15, 10))
sns.heatmap(
    numeric_data.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 5. Data Cleaning
# ============================================================

# Remove unnecessary columns

unnecessary_columns = ["Unnamed: 0", "id"]

train = train.drop(
    columns=unnecessary_columns,
    errors="ignore"
)

test = test.drop(
    columns=unnecessary_columns,
    errors="ignore"
)

# Check missing target values

if train["satisfaction"].isnull().any():
    raise ValueError("Missing values found in training target!")

if test["satisfaction"].isnull().any():
    raise ValueError("Missing values found in testing target!")


# ============================================================
# 6. Separate Features and Target
# ============================================================

X = train.drop("satisfaction", axis=1)

y = train["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
})

if y.isnull().any():
    raise ValueError("Unexpected training target values!")

# Prepare test dataset

X_test = test.drop("satisfaction", axis=1)

y_test = test["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
})

if y_test.isnull().any():
    raise ValueError("Unexpected testing target values!")

# Check feature consistency

if list(X.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match!")

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# 7. Train and Validation Split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)


# ============================================================
# 8. Feature Preprocessing
# ============================================================

# Identify numerical and categorical columns

numerical_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()


# Numerical Pipeline

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Pipeline

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine Preprocessing

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# ============================================================
# 9. Model Evaluation Function
# ============================================================

def run_model(model, X_train, y_train, X_val, y_val):

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train Model

    pipeline.fit(X_train, y_train)

    # Predictions

    y_pred = pipeline.predict(X_val)

    # Prediction Probabilities

    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    # Evaluation Metrics

    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val, y_pred_proba
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    print("\nClassification Report:")

    print(classification_report(
        y_val,
        y_pred,
        zero_division=0
    ))

    print("\nConfusion Matrix:")

    cm = confusion_matrix(y_val, y_pred)

    print(cm)

    return pipeline, accuracy, precision, recall, f1, roc_auc


# ============================================================
# 10. Train Machine Learning Models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    pipeline, accuracy, precision, recall, f1, roc_auc = run_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    trained_models[name] = pipeline

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc

    })


# ============================================================
# 11. Model Comparison
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# Select Best Model using ROC AUC

best_model_name = results_df.sort_values(
    by="ROC AUC",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ============================================================
# 12. Model Comparison Visualization
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

results_plot = results_df.set_index("Model")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xlabel("Models")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# 13. Final Testing on Test Dataset
# ============================================================

print("\n" + "=" * 60)
print("FINAL TESTING RESULTS")
print("=" * 60)

# Predictions

y_test_pred = best_model.predict(X_test)

y_test_proba = best_model.predict_proba(X_test)[:, 1]


# Evaluation

test_accuracy = accuracy_score(
    y_test, y_test_pred
)

test_precision = precision_score(
    y_test, y_test_pred, zero_division=0
)

test_recall = recall_score(
    y_test, y_test_pred, zero_division=0
)

test_f1 = f1_score(
    y_test, y_test_pred, zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test, y_test_proba
)


print("\nBest Model:", best_model_name)

print("\nTest Accuracy:", round(test_accuracy, 4))

print("Test Precision:", round(test_precision, 4))

print("Test Recall:", round(test_recall, 4))

print("Test F1 Score:", round(test_f1, 4))

print("Test ROC AUC:", round(test_roc_auc, 4))


# Classification Report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_test_pred,
    zero_division=0
))


# Confusion Matrix

cm = confusion_matrix(y_test, y_test_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 14. Confusion Matrix Visualization
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Satisfied", "Satisfied"],
    yticklabels=["Not Satisfied", "Satisfied"]
)

plt.title("Final Model Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.tight_layout()
plt.show()


# ============================================================
# 15. Save Best Model
# ============================================================

joblib.dump(
    best_model,
    "best_passenger_model.pkl"
)

print("\nBest Model Saved Successfully!")

print("\nSaved File: best_passenger_model.pkl")


# ============================================================
# 16. Save Model Comparison Results
# ============================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nModel Comparison Results Saved Successfully!")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
# ============================================================

# ============================================================
# 17. FEATURE IMPORTANCE ANALYSIS
# ============================================================

# Get trained Random Forest model
rf_model = best_model.named_steps["classifier"]

# Get preprocessed feature names
feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance values
importance_values = rf_model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort features
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display Top 15 Features
print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15).to_string(index=False))

# Visualization
plt.figure(figsize=(12, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Passenger Satisfaction Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Save Feature Importance
feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature Importance Saved Successfully!")
