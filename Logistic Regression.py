import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

cd/content/drive/MyDrive/Dataset

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataset=pd.read_csv('bank.csv')

dataset.isnull().sum()

dataset.shape

dataset["Prediction"]=dataset["Prediction"].map({"yes":1,"no":0})
dataset

x = dataset.loc[0:4119,['age','duration','campaign','pdays','previous','nr.employed','cons.conf.idx','cons.price.idx','emp.var.rate']]
y = dataset.loc[0:4119,['Prediction']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()
logreg.fit(x_train,y_train)
Y_pred_test = logreg.predict(x_test)

Y_pred_train = logreg.predict(x_train)

sns.regplot(x=y_test,y=Y_pred_test,ci=None,color='Green')

from sklearn.metrics import accuracy_score,precision_score,recall_score

#Test Data

print('Precision_Score_test = ', precision_score(y_test, Y_pred_test)*100)
print('Recall_Score_test = ', recall_score(y_test, Y_pred_test)*100)
print('Accuracy_Score_test = ', accuracy_score(y_test, Y_pred_test)*100)

#Train Data

print('Precision_Score_train = ', precision_score(y_train, Y_pred_train)*100)
print('Recall_Score_train = ', recall_score(y_train, Y_pred_train)*100)
print('Accuracy_Score_train = ', accuracy_score(y_train, Y_pred_train)*100)
logreg.score(x_train,y_train)
