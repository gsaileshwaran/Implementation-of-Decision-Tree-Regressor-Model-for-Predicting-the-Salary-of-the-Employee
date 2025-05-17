# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries for data handling, model training, and evaluation.
2. Read and preprocess the dataset, converting categorical data to numerical using Label Encoding.
3. Split the data into training and testing sets for model evaluation.
4. Train the Decision Tree Classifier using the training data and make predictions on the test set.
5. Evaluate the model using accuracy, mean squared error, and R² score, then make a sample prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Saileshwaran Ganesan
RegisterNumber:  212224230237
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
```

```
data=pd.read_csv("/content/Salary.csv")
print(data.head())
```

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print(data.head())
```

```
x=data[["Position","Level"]]
y=data["Salary"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
print("mean squared error:",mean_squared_error(y_test,y_pred))
```

```
sc=r2_score(y_pred,y_test)
print("r2: ",sc)
print(model.predict([[5,6]]))
```

## Output:
DISPLAY TOP ROWS

![image](https://github.com/user-attachments/assets/dc4911f0-29f5-4c96-bf10-77d16452152c)

AFTER LABEL ENCODING

![image](https://github.com/user-attachments/assets/51dc6dc0-0179-4761-aa3e-dc80c6d4e927)

ACCURACY AND MSE

![image](https://github.com/user-attachments/assets/33af32f6-f84e-4f6c-b5ef-39cfb94d3bdd)

PREDICTION

![image](https://github.com/user-attachments/assets/7abce47d-b9a1-4a13-9235-3b4989d25537)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
