from google.colab import drive
drive.mount('/content/drive')

cd/content/drive/My Drive/dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Heart.csv")
data

data.info()

data.describe()

data.isnull().sum()

from sklearn.neighbors import KNeighborsClassifier

X=data[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']]
y=data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score=f1_score(y_test, y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1_score)

print(classifier.score(X_test, y_test)*100)

