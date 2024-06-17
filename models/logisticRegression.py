import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset import *

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred1 = model.predict(x_test)


print("Single Usage model :: \n")
print("Mean squared error ::", mean_squared_error(y_test, y_pred1))
print("R2 score ::", r2_score(y_test, y_pred1))
print("Accuracy score ::", accuracy_score(y_test, y_pred1))


adaClassifier = AdaBoostClassifier(estimator=model, random_state=42)
adaClassifier.fit(x_train, y_train)
y_pred2 = adaClassifier.predict(x_test)

print("\nAda Boost Classifier model :: \n")
print("Mean squared error ::", mean_squared_error(y_test, y_pred2))
print("R2 score ::", r2_score(y_test, y_pred2))
print("Accuracy score ::", accuracy_score(y_test, y_pred2))


grad = GradientBoostingClassifier(init=model, random_state=42)
grad.fit(x_train, y_train)
y_pred3 = grad.predict(x_test)

print("\nGradient Boost Classifier model :: \n")
print("Mean squared error ::", mean_squared_error(y_test, y_pred3))
print("R2 score ::", r2_score(y_test, y_pred3))
print("Accuracy score ::", accuracy_score(y_test, y_pred3))