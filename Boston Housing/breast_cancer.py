import numpy as np
import pandas as pd
import matplotlib as plt

#import the data
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

print(breast_cancer.data.shape)
print(breast_cancer.feature_names)

#convert the data from nd array to dataframe and add feature names to the data
data = pd.DataFrame(breast_cancer.data)
data.columns = breast_cancer.feature_names
print(data.head(10))

breast_cancer.target.shape
data['diagnostic'] = breast_cancer.target
print(data.head())
#description and info of the dataset
print(data.describe())
print(data.info())
