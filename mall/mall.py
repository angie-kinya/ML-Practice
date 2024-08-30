import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('mall/data/Mall_Customers.csv')
print(df.head())

# Basic information about the data
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# EDA
# Visualize the distribution of age, annual income, and spending score
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sb.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sb.histplot(df['Annual Income (k$)'], kde=True, color='red')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sb.histplot(df['Spending Score (1-100)'], kde=True, color='purple')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()

# DATA PROCESSING
# Features selection
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Use the Elbow method to find the optimal number of clusters
wcss = [] # within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# From the plot, the optimal number of clusters is 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the original DataFrame
df['Cluster'] = clusters
print(df.head())

# Plot the clusters
plt.figure(figsize=(10, 6))
sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()