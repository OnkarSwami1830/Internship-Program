import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_excel("creditcard.csv.xlsx")

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION EDA")
print("=" * 60)

# ============================================
# BASIC INFORMATION
# ============================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============================================
# TRANSACTION ANALYSIS
# ============================================

total_transactions = len(df)
fraud_transactions = df[df['Class'] == 1].shape[0]
genuine_transactions = df[df['Class'] == 0].shape[0]

fraud_percentage = (fraud_transactions / total_transactions) * 100

print("\n==============================")
print("TRANSACTION ANALYSIS")
print("==============================")

print("Total Transactions :", total_transactions)
print("Genuine Transactions :", genuine_transactions)
print("Fraud Transactions :", fraud_transactions)
print("Fraud Percentage :", round(fraud_percentage, 4), "%")

# ============================================
# AMOUNT ANALYSIS
# ============================================

print("\n==============================")
print("AMOUNT ANALYSIS")
print("==============================")

print("Maximum Transaction :", df['Amount'].max())
print("Minimum Transaction :", df['Amount'].min())
print("Average Transaction :", df['Amount'].mean())
print("Median Transaction :", df['Amount'].median())

# ============================================
# TIME ANALYSIS
# ============================================

df['Hour'] = (df['Time'] // 3600).astype(int)

fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()

print("\n==============================")
print("TIME ANALYSIS")
print("==============================")

print(fraud_by_hour)

print("\nPeak Fraud Hour :", fraud_by_hour.idxmax())

# ============================================
# FEATURE SCALING
# ============================================

print("\n==============================")
print("FEATURE SCALING")
print("==============================")

standard_scaler = StandardScaler()

df['Amount_Standard'] = standard_scaler.fit_transform(
    df[['Amount']]
)

minmax_scaler = MinMaxScaler()

df['Amount_MinMax'] = minmax_scaler.fit_transform(
    df[['Amount']]
)

print("\nStandard Scaled Amount")
print(df['Amount_Standard'].head())

print("\nMinMax Scaled Amount")
print(df['Amount_MinMax'].head())

print("\nStandardScaler is generally preferred for Credit Card Fraud Detection.")

# ============================================
# DATA ENCODING
# ============================================

print("\n==============================")
print("DATA ENCODING")
print("==============================")

categorical_columns = df.select_dtypes(include=['object']).columns

if len(categorical_columns) == 0:
    print("No categorical columns found.")
else:

    le = LabelEncoder()

    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=categorical_columns)

    print("Encoding Completed.")

# ============================================
# CLASS IMBALANCE ANALYSIS
# ============================================

print("\n==============================")
print("CLASS IMBALANCE ANALYSIS")
print("==============================")

print(df['Class'].value_counts())

print("\nThis dataset is highly imbalanced.")
print("Most transactions are Genuine.")
print("Very few transactions are Fraud.")

# ============================================
# COUNT PLOT
# ============================================

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Genuine")
plt.show()

# ============================================
# HISTOGRAM
# ============================================

plt.figure(figsize=(8,5))
plt.hist(df['Amount'], bins=50)
plt.title("Transaction Amount Histogram")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# ============================================
# BOX PLOT
# ============================================

plt.figure(figsize=(6,5))
sns.boxplot(y=df['Amount'])
plt.title("Transaction Amount Box Plot")
plt.show()

# ============================================
# PIE CHART
# ============================================

plt.figure(figsize=(6,6))

plt.pie(
    df['Class'].value_counts(),
    labels=['Genuine', 'Fraud'],
    autopct='%1.2f%%',
    explode=(0, 0.15),
    shadow=True
)

plt.title("Class Distribution")
plt.show()

# ============================================
# DISTRIBUTION PLOT
# ============================================

plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], kde=True)
plt.title("Amount Distribution")
plt.show()

# ============================================
# FRAUD BY HOUR
# ============================================

plt.figure(figsize=(10,5))
fraud_by_hour.plot(kind='bar')
plt.title("Fraud Transactions By Hour")
plt.xlabel("Hour")
plt.ylabel("Fraud Count")
plt.show()

# ============================================
# CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ============================================
# FINAL SUMMARY
# ============================================

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")

print("Total Transactions :", total_transactions)
print("Fraud Transactions :", fraud_transactions)
print("Fraud Percentage :", round(fraud_percentage, 4), "%")
print("Maximum Amount :", df['Amount'].max())
print("Minimum Amount :", df['Amount'].min())
print("Average Amount :", round(df['Amount'].mean(), 2))
print("Median Amount :", df['Amount'].median())
print("Peak Fraud Hour :", fraud_by_hour.idxmax())

print("\nEDA COMPLETED SUCCESSFULLY")