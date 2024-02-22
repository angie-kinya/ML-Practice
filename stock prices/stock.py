import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/tesla.csv')
df.head()
df.shape
df.describe()
df.info()

#EDA
plt.figure(figsize =(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close Price.', fontsize = 15)
plt.ylabel('Price in dollars.')
plt.show()

#check is each entry in Close is the same as Adj Close
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1) #drop redundant data

#check for null values
df.isnull().sum()

features = ['Open', 'High', 'Low', 'Close', 'Volume']

#plot the distribution
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.displot(df[col])
    plt.show()

#check for outliers
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(df[col])
    plt.show()

#FEATURE ENGINEERING
#get insights from date
splitted = df['Date'].str.split('/', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()

df['is_quater_end'] = np.where(df['month']%3==0,1,0)
df.head()
