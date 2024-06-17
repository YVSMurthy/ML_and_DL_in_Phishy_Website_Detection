import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset import *

labels = ["Legitimate", "Phishy"]

# model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
# model = LinearSVC(C = 10, loss='squared_hinge', random_state=42)
model = NuSVC(gamma='scale', kernel='rbf', nu=0.3, random_state=42)

# Normal SVM Model
model.fit(x_train, y_train)
y_pred1 = model.predict(x_test)

print("Single Usage model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred1)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred1)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Single Usage")
plt.show()


# Logistic regression with AdaBoost classifier
adaClassifier = AdaBoostClassifier(estimator=model, algorithm='SAMME', random_state=42)
adaClassifier.fit(x_train, y_train)
y_pred2 = adaClassifier.predict(x_test)

print("\nAdaBoost Classifier model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred2)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred2))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred2)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("AdaBoost Classifier")
plt.show()


gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

svm_predictions = model.predict(x_train)
# Fit Gradient Boosting Classifier on SVM predictions
gb.fit(x_train, svm_predictions)

y_pred3 = (gb.predict(x_test) > 0.5).astype('int32')

print("\GradientBoost Classifier model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred3)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred3))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred3)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("GradientBoost Classifier")
plt.show()