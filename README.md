# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages such as pandas, numpy, matplotlib.
2. Find the head and tail using df.head() and df.tail() method.
3. Display the arrays values of X, Y and find the regression values and display the values of y_pred and y_test.
4. Display the training set graph and the test set graph.
5. Display the values of MSE, MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BARATH S
RegisterNumber:  212222230018
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()

df.tail()

x = df.iloc[:,:-1].values
x

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

y_pred
y_test

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'orange')
plt.title("Hours vs Scores (Traning Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,regressor.predict(x_test),color = 'orange')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)v

```

## Output:
## df.head()
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/head.png)

## df.tail()
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/tail.png)

## Array value of X
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/x.png)

## Array value of Y
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/y.png)

## Values of Y prediction
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ypred.png)

## Array values of Y test
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ytest.png)

## Training Set Graph
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/train.png)

## Test Set Graph
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/test.png)

## Values of MSE, MAE, RMSE
![image](https://github.com/barathsubramani/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/mse.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
