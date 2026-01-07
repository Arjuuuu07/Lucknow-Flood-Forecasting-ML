import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load processed dataset
df = pd.read_csv('newfile.csv')

#  Data Cleaning 

drop_cols = [c for c in df.columns if 'lat' in c or 'lon' in c or 'date' in c]
X = df.drop(columns=drop_cols + ['flood'])
y = df['flood']

# Splitting 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

#  Preprocessing 
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_processed = scaler.transform(imputer.transform(X_test))

#  Model Training 

model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
model.fit(X_train_processed, y_train)

# Detailed Evaluation
y_pred = model.predict(X_test_processed)

# 1. Classification Report
report = classification_report(y_test, y_pred, output_dict=True)

print("\n MODEL PERFORMANCE: FLOOD DETECTION ")

print(f"Recall:    {report['1']['recall']:.2f}")
print(f"F1-Score:  {report['1']['f1-score']:.2f}")

# 2. Confusion Matrix 

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n ERROR ANALYSIS ")
print(f"True Positives (Correct Alerts): {tp}")
print(f"True Negatives (Correct Quiet Days): {tn}")
print(f"False Positives (False Alarms): {fp}")
print(f"False Negatives (MISSED FLOODS): {fn}")



# Feature Impact
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
}).sort_values(by='Importance', ascending=False)

print("\n TOP 5 DRIVERS OF LUCKNOW FLOODS ")
print(importance.head(5))