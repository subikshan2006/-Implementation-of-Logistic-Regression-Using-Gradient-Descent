# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.


## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:- subikshan.p
RegisterNumber: 212223240161


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("Social_Network_Ads.csv")
x=datasets.iloc[:,[2,3]].values
y=datasets.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()

from sklearn.linear_model import LogisticRegression
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
recall_sensitivity=metrics.recall_score(Y_test,Y_pred,pos_label=1)
recall_sensiticity=metrics.recall_score(Y_test,Y_pred,pos_label=0)
recall_sensitivity,recall_sensiticity

from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
  plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)
plt.title("Logistic Regression(Training set")
plt.xlabel("AGE")
plt.ylabel("ESTIMATED SALARY")
plt.legend()
plt.show()
```

## Output:
### Array value of X:

![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/04571f0b-63bb-4f34-b7d0-55155c6f741c)


### Array value of Y:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/aa6b9eb2-d096-42c6-813f-b0af0e787713)


### Exam 1-Score graph:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/886bc2c1-867d-43a4-b806-263091680403)


### Sigmoid function graph:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/bf0c020e-c63d-4049-91b4-182aff6ede8f)

### X_Train_grad value:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/8f5adc42-6422-4806-a0fd-63087efb4db8)

### Y_Train_grad value:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/0120579a-f68d-4eb2-8ab6-781386b5747c)


### Print res.X:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/b024a2e4-3910-46eb-908d-5dfbc7af05d8)


### Decision boundary-gragh for exam score:
![Screenshot 2024-05-12 112718](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/ced8add3-2616-420a-9a74-0c8142bf3959)


### Probability value:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/6fe2c242-703e-4b0f-abe6-db8c29b2281a)


### Prediction value of mean:
![image](https://github.com/subikshan2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139841805/d79df38e-2c47-4871-b4ad-1ecead5499e5)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
