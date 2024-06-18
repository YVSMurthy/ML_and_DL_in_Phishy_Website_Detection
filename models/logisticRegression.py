import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset import *

labels = ["Legitimate", "Phishy"]

model = LogisticRegression(solver='liblinear', max_iter=1000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],    # Regularization parameter
    'penalty': ['l1', 'l2'],               # Regularization penalty
}

# 'solver': ['liblinear', 'newton-cg', 'sag', 'saga']  # Solvers to test

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# # Fit GridSearchCV using the training data
# grid_search.fit(x_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_

# print("Best Parameters:", grid_search.best_params_)

# Normal Logistic Regression Model
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
y_pred_uk = model.predict(x_test_uk)

print("Single Usage model :: \n")
print("Testing Accuracy score ::", round(accuracy_score(y_test, y_pred_test)*100, 5))
print("Testing Classification report ::\n", classification_report(y_test, y_pred_test))

print("\nUnknown Testing Accuracy score ::", round(accuracy_score(y_test_uk, y_pred_uk)*100, 5))
print("Unknown Testing Classification report ::\n", classification_report(y_test_uk, y_pred_uk))

# # Confusion Matrix
# # conf_matrix = confusion_matrix(y_test, y_pred1)
# # print(conf_matrix)

# # # Visualize the confusion matrix
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.title("Single Usage")
# # plt.show()

# Logistic regression with AdaBoost classifier
adaClassifier = AdaBoostClassifier(estimator=model, random_state=42, algorithm='SAMME')
adaClassifier.fit(x_train, y_train)
y_pred2 = adaClassifier.predict(x_test)
y_pred2_uk = adaClassifier.predict(x_test_uk)

print("\nAdaBoost Classifier model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred2)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred2))

print("\nAdaBoost Classifier model unknown :: \n")
print("Accuracy score ::", round(accuracy_score(y_test_uk, y_pred2_uk)*100, 5))
print("Classification report ::\n", classification_report(y_test_uk, y_pred2_uk))

# # Confusion Matrix
# # conf_matrix = confusion_matrix(y_test, y_pred2)

# # # Visualize the confusion matrix
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.title("AdaBoost Classifier")
# # plt.show()

# # Logistic Regression with GradientBoost classifier
grad = GradientBoostingClassifier(init=model, random_state=42)
grad.fit(x_train, y_train)
y_pred3 = grad.predict(x_test)
y_pred3_uk = grad.predict(x_test_uk)

print("\nGradientBoost Classifier model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred3)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred3))

print("\nGradientBoost Classifier model unknown :: \n")
print("Accuracy score ::", round(accuracy_score(y_test_uk, y_pred3_uk)*100, 5))
print("Classification report ::\n", classification_report(y_test_uk, y_pred3_uk))

# # Confusion Matrix
# # conf_matrix = confusion_matrix(y_test, y_pred3)

# # # Visualize the confusion matrix
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.title("GradientBoost classifier")
# # plt.show()