import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\forcast\weatherforcast\dailyskies\weather.csv")
df.dropna(inplace=True)

# Add a synthetic date column (optional)
from datetime import datetime, timedelta
df['date'] = [datetime.today() - timedelta(days=i) for i in range(len(df))]

# Encode WindGustDir
le = LabelEncoder()
df['WindGustDir'] = le.fit_transform(df['WindGustDir'].astype(str))

# --------- CLASSIFICATION (RainTomorrow) ---------
features = ['MinTemp', 'MaxTemp', 'Temp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']
X_cls = df[features]
y_cls = LabelEncoder().fit_transform(df['RainTomorrow'].astype(str))

# Balance data with SMOTE
sm = SMOTE(random_state=42)
X_cls_res, y_cls_res = sm.fit_resample(X_cls, y_cls)

# Split data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls_res, y_cls_res, test_size=0.2, random_state=42)

# RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_cls, y_train_cls)
print("RandomForestClassifier:")
print(classification_report(y_test_cls, rf_clf.predict(X_test_cls)))

# Logistic Regression
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train_cls, y_train_cls)
print("LogisticRegression:")
print(classification_report(y_test_cls, lr_clf.predict(X_test_cls)))

# XGBClassifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train_cls, y_train_cls)
print("XGBClassifier:")
print(classification_report(y_test_cls, xgb_clf.predict(X_test_cls)))

# Save best classifier model
joblib.dump(rf_clf, 'rain_model.joblib')

# --------- REGRESSION (Temp & Humidity) ---------
scaler = StandardScaler()

# Temp prediction
X_temp = df[features]
y_temp = df['Temp'].shift(-1).fillna(method='ffill')  # next day's temp
X_temp_scaled = scaler.fit_transform(X_temp)
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp_scaled, y_temp, test_size=0.2, random_state=42)

# Gradient Boosting Regressor
gb_temp = GradientBoostingRegressor()
gb_temp.fit(X_train_temp, y_train_temp)
y_pred_temp = gb_temp.predict(X_test_temp)
print("\nGradientBoostingRegressor (Temp):")
print("MSE:", mean_squared_error(y_test_temp, y_pred_temp))
print("R2:", r2_score(y_test_temp, y_pred_temp))


# XGBRegressor for Humidity
y_hum = df['Humidity'].shift(-1).fillna(method='ffill')
X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(X_temp_scaled, y_hum, test_size=0.2, random_state=42)
xgb_hum = XGBRegressor()
xgb_hum.fit(X_train_hum, y_train_hum)
y_pred_hum = xgb_hum.predict(X_test_hum)
print("\nXGBRegressor (Humidity):")
print("MSE:", mean_squared_error(y_test_hum, y_pred_hum))
print("R2:", r2_score(y_test_hum, y_pred_hum))
import pickle

# Save models
with open('rain_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

with open('temp_model.pkl', 'wb') as f:
    pickle.dump(gb_temp, f)

with open('humidity_model.pkl', 'wb') as f:
    pickle.dump(xgb_hum, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('wind_dir_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
