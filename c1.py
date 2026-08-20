# ==============================
# Data Preprocessing & Feature Engineering
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_excel("creditcard.csv.xlsx")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info")
print(df.info())

# Statistical summary
print("\nStatistical Summary")
print(df.describe())

# ===================================================
# 1. Missing Value Analysis
# ===================================================

print("\nMissing Values")
print(df.isnull().sum())

# Missing value percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\nMissing Percentage")
print(missing_percent)

# Fill missing numerical values with median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after handling")
print(df.isnull().sum())

# ===================================================
# 2. Duplicate Data Analysis
# ===================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

# Remove duplicates
df = df.drop_duplicates()

print("Duplicates after removal:", df.duplicated().sum())

# ===================================================
# 3. Outlier Detection using IQR
# ===================================================

print("\nOutlier Detection")

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(col, ":", len(outliers))

# ===================================================
# Boxplots
# ===================================================

for col in numeric_cols:

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot of {col}")
    plt.show()

# ===================================================
# Remove Outliers (Optional)
# ===================================================

clean_df = df.copy()

for col in numeric_cols:

    Q1 = clean_df[col].quantile(0.25)
    Q3 = clean_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean_df = clean_df[
        (clean_df[col] >= lower) &
        (clean_df[col] <= upper)
    ]

print("\nOriginal Shape:", df.shape)
print("Shape After Removing Outliers:", clean_df.shape)

# ===================================================
# 4. Feature Correlation
# ===================================================

corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix")
print(corr)

# Heatmap

plt.figure(figsize=(14,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.show()

# ===================================================
# Highly Correlated Features
# ===================================================

threshold = 0.80

high_corr = []

for i in range(len(corr.columns)):
    for j in range(i):

        if abs(corr.iloc[i, j]) > threshold:

            high_corr.append((
                corr.columns[i],
                corr.columns[j],
                corr.iloc[i, j]
            ))

print("\nHighly Correlated Features")

for item in high_corr:
    print(item)

# ===================================================
# Features Affecting Fraud
# ===================================================

if "Class" in df.columns:

    print("\nCorrelation with Fraud (Class)\n")

    fraud_corr = corr["Class"].sort_values(ascending=False)

    print(fraud_corr)

    plt.figure(figsize=(8,6))

    fraud_corr.plot(kind="bar")

    plt.title("Feature Importance Based on Correlation")

    plt.ylabel("Correlation")

    plt.show()

else:
    print("\nTarget column 'Class' not found.")

print("\nData preprocessing completed successfully.")