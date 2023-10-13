import pandas as pd
import sklearn.datasets as datas
from sklearn import tree
from sklearn.model_selection import train_test_split

irisdata=datas.load_iris()
irisdata

type(irisdata)

irisdata.keys()

irisdata.data

irisdata.target

irisdata.target_names

irisIndData=irisdata.data

#convert the object into a pandas dataframe
irisIndData=pd.DataFrame(irisdata.data)
irisIndData.head(5)

#display feature names in the datasset
irisdata.feature_names

#create list with all the column names
irisIndData.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth']
irisIndData.head(5)

irisIndTrain, irisIndTest, irisDepTrain, irisDepTest =train_test_split(irisIndData,irisdata.target,test_size=0.2,random_state=0)

dt=tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')

modl=dt.fit(irisIndTrain,irisDepTrain)

predictedData=modl.predict(irisIndTest)
predictedData

import graphviz

dot_data=tree.export_graphviz(dt, out_file=None,feature_names=irisdata.feature_names, class_names=irisdata.target_names,  filled=True, rounded=True, special_characters=True)

graph=graphviz.Source(dot_data)
graph
