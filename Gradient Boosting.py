import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("cirrhosis.csv")

dataset.head()
dataset.isnull().sum()
dataset = dataset.dropna()
dataset.shape
dataset.corr()

x =dataset[["N_Days", "Age", "Bilirubin", 'Cholesterol', "Albumin", "Copper",'Alk_Phos', 'SGOT','Tryglicerides', 'Platelets',  'Prothrombin' ]] 
y = dataset["Stage"]

dataset.columns

plt.plot(dataset['Cholesterol'],dataset['Stage'])

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25,random_state= 2)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy,precision,recall
grad_model = GradientBoostingClassifier()

grad_model.fit(x_train,y_train) 

print('the accuracy :',accuracy)
print('the precision score :',precision)
print('the recall score :',recall)
print(f1_score(y_test,y_pred,average='macro'))
print(confusion_matrix(y_test,y_pred))

RocCurveDisplay.from_predictions(y_test,y_pred,pos_label=2)

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
