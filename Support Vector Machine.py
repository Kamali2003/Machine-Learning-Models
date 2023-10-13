import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

from google.colab import drive 
drive.mount('/content/drive') 

cd/content/drive/MyDrive/Dataset 

dataset=pd.read_csv("heart.csv") 

dataset.isnull().sum() 

from sklearn.model_selection import train_test_split 

X=dataset.loc[0:918,['Age','RestingBP’,'Cholesterol', 'FastingBS',  'MaxHR','Oldpeak']] 
Y=dataset.loc[0:918,'HeartDisease'] 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0, train_size=0.75) 

from sklearn.svm import SVC 
SVCClf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False) 
SVCClf.fit(X_train,Y_train)#train the model using the training sets 

Y_pred=SVCClf.predict(X_test)#predict the response for test dataset 

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score 

cm = confusion_matrix(Y_test, Y_pred) 
ac = accuracy_score(Y_test,Y_pred) 
pc= precision_score(Y_test,Y_pred) 

print(cm)
print(ac)
print(pc) 

class_names=[0,1] # name  of classes 

fig, ax = plt.subplots() 

tick_marks = np.arange(len(class_names)) 

plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 

ax.xaxis.set_label_position("top") 

plt.tight_layout() 

plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 
