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

df = pd.read_csv('stock prices/data/tesla.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

#EDA
plt.figure(figsize =(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close Price.', fontsize = 15)
plt.ylabel('Price in dollars.')
plt.show()

#check for redundant data
print(df[df['Close'] == df['Adj Close']].shape)
df = df.drop(['Adj Close'], axis=1) #drop redundant data

#check for null values
print(df.isnull().sum())

features = ['Open', 'High', 'Low', 'Close', 'Volume']

#plot the distribution
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.histplot(df[col], kde=True)
plt.show()

#check for outliers
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(df[col])
plt.show()

#FEATURE ENGINEERING
#get insights from date
# Split the 'Date' column
splitted = df['Date'].str.split('-', expand=True)

# Assign day, month, and year to DataFrame columns
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Print the DataFrame to check the changes
print(df.head())

df['is_quater_end'] = np.where(df['month']%3==0,1,0)
print(df.head())

# Ensure the 'Date' column is in the correct format
df['Date'] = pd.to_datetime(df['Date'])
# Extract the year, month, and day from the 'Date' column
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot(kind='bar')
plt.show()

print(df.groupby('is_quater_end').mean())

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
#check if target is balanced
plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

#TRAINING AND TESTING
#split the data into features(x) and targets(y)
X = df[['open-close', 'low-high', 'is_quater_end']]
y = df['target']

#split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#apply logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, y_pred_log_reg))

#apply SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))

#apply XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", metrics.accuracy_score(y_test, y_pred_xgb))