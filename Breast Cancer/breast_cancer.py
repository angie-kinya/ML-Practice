import numpy as np
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

#import the data
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

print(breast_cancer.data.shape)
print(breast_cancer.feature_names)

#convert the data from nd array to dataframe and add feature names to the data
data = pd.DataFrame(breast_cancer.data)
data.columns = breast_cancer.feature_names
print(data.head(10))

#description and info of the dataset
print(data.describe())
print(data.info())

#input data
x = breast_cancer.data
#output data
y = breast_cancer.target

#split data into training and testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.2, random_state=0)

print("xtrain shape : ", xtrain.shape)
print("xtest shape : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape : ", ytest.shape)

#fit logistic regression to train the model
from sklearn.linear_model import LogisticRegression
# Instantiate the Logistic Regression model
classifier = LogisticRegression(max_iter=10000)
# Train the model using the training data
classifier.fit(xtrain, ytrain)
# Predict the test set results
y_pred = classifier.predict(xtest)

#evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(ytest, y_pred)
print("Accuracy: ", accuracy)

print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred))

print("Classification Report:\n", classification_report(ytest, y_pred))

#plot the confusion matrix
sb.heatmap(confusion_matrix(ytest, y_pred), annot=True, fmt="d", cmap="coolwarm")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()