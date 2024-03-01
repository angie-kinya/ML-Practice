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

#fit linear regression to train the model
from sklearn.linear_model import LinearRegression
# Instantiate the LinearRegression model
regressor = LinearRegression()
# Train the model using the training data
regressor.fit(xtrain, ytrain)
# Predict the test set results
y_pred = regressor.predict(xtest)

#plot the prediction
plt.scatter(ytest, y_pred, c = "purple")
plt.xlabel("Diagnosis")
plt.ylabel("Predicted Value")
plt.title("Diagnosis vs Predicted Value : Linear Regression")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(ytest, y_pred)
mae = mean_absolute_error(ytest,y_pred)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)
