import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset import *

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred1 = rf_clf.predict(x_test)
y_pred2 = rf_clf.predict(x_test_uk)

# Evaluate the classifier
print("\nScores of known dataset :: \n")
accuracy = accuracy_score(y_test, y_pred1)
class_report = classification_report(y_test, y_pred1)
print("Accuracy :: ", round(accuracy*100, 5))
print("Classification report ::",class_report, sep="\n")

print("\nScores of unknown dataset :: \n")
accuracy = accuracy_score(y_test_uk, y_pred2)
class_report = classification_report(y_test_uk, y_pred2)
print("Accuracy :: ", round(accuracy*100, 5))
print("Classification report ::",class_report, sep="\n")
