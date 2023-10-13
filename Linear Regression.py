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

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train) 
pred=regressor.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

mse=mean_squared_error(y_test,pred)
r2=r2_score(y_test,pred)
print(mse)
print(r2)

plt.plot(x_test,pred,color='Red',marker='o')
