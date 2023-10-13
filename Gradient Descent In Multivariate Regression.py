from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cd/content/drive/MyDrive/dataset

%pip install mlxtend --upgrade

from mlxtend.evaluate import bias_variance_decomp

from sklearn.linear_model import LinearRegression
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp

pwd

dataset=pd.read_csv("house data.csv")
dataset

dataset.columns

import statsmodels.api as sm
import statsmodels.formula.api as smf

x=dataset.drop('id', axis=1,inplace=True)
x

y=dataset['price']
x=dataset.drop('price', axis=1)
x=dataset.drop('date', axis=1)

x=sm.add_constant(x, prepend=False)

x.columns

model=sm.OLS(y, x.astype(float)).fit()
predictions=model.predict(x)
print(model.params)

model.summary()

model12=smf.ols(formula='price~waterfront',data=x).fit()
model12.summary()

model13=smf.ols(formula='price~bedrooms',data=x).fit()
model13.summary()

from mpl_toolkits.mplot3d import Axes3D
def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error
def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1/m) * np.dot(X.T, error))
        cost_array[i] = cost
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()
def run():
    data = pd.read_csv('house data.csv')
    X = data[['bathrooms', 'bedrooms']]
    y = data['price']
    X = (X - X.mean()) / X.std()
    X = np.c_[np.ones(X.shape[0]), X] 
    alpha = 0.01
    iterations = 1000
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)
    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)
    plotChart(iterations, cost_num)
    final_cost, _ = cost_function(X, y, theta)
    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))

if  __name__ == "__main__":
    run()




