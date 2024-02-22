import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/netflix_titles.csv')
print(df.head())

df.shape
df.info()
df.describe().T

df['type'].value_counts()
df['rating'].value_counts()

plt.pie(df['type'].value_counts().values, autopct='%1.1f%%')
plt.show()

ints = []
objects = []

for col in df.columns:
    if df[col].dtype == int:
        ints.append(col)
    else:
        objects.append(col)

objects.remove('show_id')
objects.remove('type')

plt.subplots(figsize=(15,30))

for i, col in enumerate(objects):
    plt.subplot(5,3,i+1)
    sb.countplot(df[col], hue=df['type'])
    plt.xticks(rotation=60)
plt.tight_layout()
plt.show()