import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
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

# building the model
model = Sequential()
model.add(Input(shape=(49,)))
model.add(Dense(units=64, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=32, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=16, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=8, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=4, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=2, activation=LeakyReLU(negative_slope=0.01)))
model.add(Dense(units=1, activation='sigmoid'))

#compiling the model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#training the model
model.fit(x_train, y_train, batch_size=32, epochs=100)

#predicting the result
y_pred = (model.predict(x_test) > 0.5).astype('int32')

#checking the accuracy 
print("\nANN model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ANN Model")
plt.show()


#boosting
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
gradientboost_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

ann_train = (model.predict(x_train) > 0.5).astype('int32')
ann_train = ann_train.ravel()

#training
adaboost_model.fit(x_train, ann_train)
gradientboost_model.fit(x_train, ann_train)

#prediction
y_pred2 = (adaboost_model.predict(x_test) > 0.5).astype('int32')
y_pred3 = (gradientboost_model.predict(x_test) > 0.5).astype('int32')

#accuracy measure
#checking the accuracy 
print("\nAdaBoost model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred2)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred2))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred2)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("AdaBoost Model")
plt.show()


#checking the accuracy 
print("\nGradientBoost model :: \n")
print("Accuracy score ::", round(accuracy_score(y_test, y_pred3)*100, 5))
print("Classification report ::\n", classification_report(y_test, y_pred3))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred3)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("GradientBoost Model")
plt.show()