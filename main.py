# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Load dataset (make sure the CSV is in the same folder)
df = pd.read_csv("HR_comma_sep.csv")

# STEP 1: BASIC EXPLORATION & CLEANING

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nShape of dataset:", df.shape)

# Info about datatypes and nulls
print("\nData Types & Nulls:")
print(df.info())

# Descriptive statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)

# 2.1 Retention vs Salary
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='salary', hue='left', palette='Set2')
plt.title('Employee Retention by Salary Level')
plt.xlabel('Salary Level')
plt.ylabel('Number of Employees')
plt.legend(title='Left Company', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 2.2 Retention vs Department
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='Department', hue='left', palette='Set1')
plt.title('Employee Retention by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.legend(title='Left Company', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2.3 Correlation Heatmap

# Prepare data for correlation (convert categorical features)
df_corr = df.copy()
df_corr['salary'] = df_corr['salary'].map({'low': 0, 'medium': 1, 'high': 2})
df_corr = pd.get_dummies(df_corr, columns=['Department'], drop_first=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prepare data
df_model = df.copy()

# Encode 'salary'
df_model['salary'] = df_model['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# One-hot encode 'Department'
df_model = pd.get_dummies(df_model, columns=['Department'], drop_first=True)

# Split into features and target
X = df_model.drop('left', axis=1)
y = df_model['left']

# Split into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("\n✅ Logistic Regression Results:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
rf_preds = rf_model.predict(X_test)

print("\n🌲 Random Forest Results:")
print("Accuracy Score:", accuracy_score(y_test, rf_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("\nClassification Report:")
print(classification_report(y_test, rf_preds))

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Plot Confusion Matrix for RF
cm = confusion_matrix(y_test, rf_preds)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Left'], yticklabels=['Stay', 'Left'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

import joblib
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")