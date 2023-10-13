from google.colab import drive
drive.mount('/content/drive')

cd/content/drive/My Drive/dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("healthcare.csv")
data

data.describe()

data.info()

data.isnull().sum()

X=data[['id','age','hypertension','heart_disease','avg_glucose_level']]
y=data['stroke']

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score=f1_score(y_test, y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1_score)
