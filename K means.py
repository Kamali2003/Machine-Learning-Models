from google.colab import drive
drive.mount('/content/drive')

cd/content/drive/My Drive/dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Heart.csv")
dataset
dataset.info()

dataset.describe()

dataset.isnull().sum()

X=dataset[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']]
y=dataset['HeartDisease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

m sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score=f1_score(y_test, y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1_score)

import sklearn.cluster
kmeans = sklearn.cluster.KMeans(n_clusters=5,init='k-means++',random_state=0).fit(X)
print("\n\n Cluster Center: \n ")
print(kmeans.cluster_centers_)
print("\n\n Lables")
print(kmeans.labels_)

X= dataset.loc[:, ['Age','HeartDisease']].values

plt.scatter(X[:, 0], X[:,1], c = kmeans.labels_, cmap= "plasma")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroid')
plt.legend()
plt.xlabel("Age")
plt.ylabel("HeartDisease")
plt.show()

