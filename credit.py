# ============================================================
# CREDIT CARD FRAUD DETECTION - MACHINE LEARNING PROJECT
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -----------------------------
# 3. BASIC DATA ANALYSIS
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# 4. CLASS DISTRIBUTION
# -----------------------------

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# Visualization

plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=df)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")

plt.show()

# -----------------------------
# 5. SEPARATE FEATURES AND TARGET
# -----------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# -----------------------------
# 6. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# -----------------------------
# 7. FEATURE SCALING
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 8. DEFINE MODELS
# -----------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        )
}

# -----------------------------
# 9. TRAIN AND EVALUATE MODELS
# -----------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    # Train model

    model.fit(X_train_scaled, y_train)

    # Prediction

    y_pred = model.predict(X_test_scaled)

    # Probability

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_probability = model.decision_function(X_test_scaled)

    # Metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix - " + name)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()


# -----------------------------
# 10. MODEL COMPARISON
# -----------------------------

results_df = pd.DataFrame(results)

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(results_df.sort_values(
    by="F1 Score",
    ascending=False
))

# -----------------------------
# 11. VISUALIZE MODEL PERFORMANCE
# -----------------------------

results_df.set_index("Algorithm")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Machine Learning Algorithm Comparison")

plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(rotation=45)

plt.legend(loc="lower right")

plt.tight_layout()

plt.show()


# -----------------------------
# 12. ROC CURVES
# -----------------------------

plt.figure(figsize=(10, 7))

for name, model in trained_models.items():

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        probability = model.decision_function(
            X_test_scaled
        )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()


# -----------------------------
# 13. FIND BEST MODEL
# -----------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n===================================")
print("BEST MODEL")
print("===================================")

print(best_model)


# -----------------------------
# 14. SAVE RESULTS
# -----------------------------

results_df.to_csv(
    "algorithm_results.csv",
    index=False
)

print("\nResults saved to algorithm_results.csv")