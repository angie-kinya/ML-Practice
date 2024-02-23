import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/wine_quality.csv')
print(df.head())

df.info()
df.describe().T

#EDA
#check for null values
print(df.isnull().sum())
df = df.drop('Id', axis=1)
#histogram to visualize continuous values
df.hist(bins=20, figsize=(10,10))
plt.show()

#count plot to show no. of data for each quality of wine
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()
#check for redundant features
plt.figure(figsize=(12,12))
sb.heatmap(df.corr() >0.7, annot=True, cbar=False)
plt.show()

#MODEL DEVELOPMENT
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
print(xtrain.shape, xtest.shape)
#normalize
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
#train the models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
for i in range(3):
    models[i].fit(xtrain, ytrain)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()
#logistic regression confusion matrix
predictions = models[i].predict(xtest)
confusion_matrix_result = metrics.confusion_matrix(ytest, predictions)
plt.show()
#print classification report for best performing model
print(metrics.classification_report(ytest,
									models[1].predict(xtest)))