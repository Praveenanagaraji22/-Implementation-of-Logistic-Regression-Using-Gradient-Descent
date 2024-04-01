# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial values for the weights (w) and bias (b).
2. Compute Predictions: Calculate the predicted probabilities using the logistic function.
3. Compute Gradient: Compute the gradient of the loss function with respect to w and b.
4. Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat steps 2-4 until convergence or a maximum number of iterations is reached.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRAVEENA N
RegisterNumber: 212222040122
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()
```
```
data.info()
```
```
data.isnull()
```
```
data.isnull().sum()
```
```
data['left'].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
data['salary']=le.fit_transform(data['salary'])
data.head()
```
```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
```
y=data['left']
y.head()
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![Screenshot 2024-04-01 155600](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/db7f1712-b196-4f0b-823f-987760094481)
![Screenshot 2024-04-01 155622](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/a25c3a29-b464-4e37-86d1-7d38d90ef8f4)
![Screenshot 2024-04-01 155644](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/2204d1d9-47b3-4b74-85e5-c17bc22bc965)
![Screenshot 2024-04-01 155700](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/2d3f7b59-1f8e-447a-83d8-2658efb07f4b)
![Screenshot 2024-04-01 155720](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/6be8daea-a39d-4233-a233-f71b5d23f098)
![Screenshot 2024-04-01 155734](https://github.com/anu-varshini11/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138969827/fac96fbf-2407-42c1-a5c5-68edf8b2e071)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
