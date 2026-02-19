# ==========================================================
# Correct Final Pipeline (Scaler after Feature Selection)
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("/Users/arun/Desktop/Unsupervised/finalpro/aa.csv")

df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
df.drop(['EmployeeNumber','EmployeeCount','Over18','StandardHours'], axis=1, inplace=True)

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ==========================================================
# STEP 1: Select Top 10 Features (RFE on raw data)
# ==========================================================

rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X, y)

selected_feature_names = X.columns[rfe.support_]
X_selected = X[selected_feature_names]

print("Selected Features:", selected_feature_names)

# ==========================================================
# STEP 2: Scale ONLY selected features
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ==========================================================
# STEP 3: Train Model
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(C=1, kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 4: Save Files
# ==========================================================

joblib.dump(model, "team8_employee_model.pkl")
joblib.dump(scaler, "team8_scaler.pkl")
joblib.dump(selected_feature_names.tolist(), "team8_feature_names.pkl")

print("✅ Correct 10-Feature Model Saved!")
