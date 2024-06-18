from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset import *

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(x_train, y_train)
y_pred_ada = ada.predict(x_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
report_ada = classification_report(y_test, y_pred_ada)
print(f"AdaBoost Accuracy: {accuracy_ada*100:.4f}")
print("AdaBoost Classification Report:")
print(report_ada)

y_pred_ada = ada.predict(x_test_uk)
accuracy_ada = accuracy_score(y_test_uk, y_pred_ada)
report_ada = classification_report(y_test_uk, y_pred_ada)
print(f"AdaBoost Accuracy unknown: {accuracy_ada*100:.4f}")
print("AdaBoost Classification Report unknown:")
print(report_ada)

# GradientBoost
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)
print(f"GradientBoost Accuracy: {accuracy_gb*100:.4f}")
print("GradientBoost Classification Report:")
print(report_gb)

y_pred_gb = gb.predict(x_test_uk)
accuracy_gb = accuracy_score(y_test_uk, y_pred_gb)
report_gb = classification_report(y_test_uk, y_pred_gb)
print(f"GradientBoost Accuracy unknown: {accuracy_gb*100:.4f}")
print("GradientBoost Classification Report unknown:")
print(report_gb)

# XGBoost
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
num_round = 100
bst = xgb.train(params, dtrain, num_round)
y_pred_xgb = bst.predict(dtest)
y_pred_xgb = [1 if pred > 0.5 else 0 for pred in y_pred_xgb]
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb*100:.4f}")
print("XGBoost Classification Report:")
print(report_xgb)

dtest = xgb.DMatrix(x_test_uk, label=y_test_uk)
y_pred_xgb = bst.predict(dtest)
y_pred_xgb = [1 if pred > 0.5 else 0 for pred in y_pred_xgb]
accuracy_xgb = accuracy_score(y_test_uk, y_pred_xgb)
report_xgb = classification_report(y_test_uk, y_pred_xgb)
print(f"XGBoost Accuracy unknown: {accuracy_xgb*100:.4f}")
print("XGBoost Classification Report unknown:")
print(report_xgb)
