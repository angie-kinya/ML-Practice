import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#load the iris dataset
df = pd.read_csv('iris/data/Iris.csv')

#split into features and labels
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

#split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#create an SVM model and train it
model = SVC()
model.fit(x_train, y_train)

#check the accuracy of the model
accuracy = model.score(x_test, y_test)

print('Test accuracy: ', accuracy)