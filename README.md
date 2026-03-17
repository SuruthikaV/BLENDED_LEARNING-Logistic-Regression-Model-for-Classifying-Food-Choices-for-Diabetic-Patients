# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate input features and target class, then normalize features and encode target labels.
2. Split the dataset into training and testing sets using stratified sampling.
3. Train a Logistic Regression model with L2 regularization on the training data and predict the test data.
4. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('food_items (1).csv')
print('Name: suruthika')
print('Reg.No: 212225040441')
print('Dataset Overview')
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]

scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000
l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)

print('Name :SURUTHIKA V')
print('Reg. No: 212225040441')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print("Name: SURUTHIKA V")
print("Reg. No: 212225040441")
```

## Output:
<img width="1075" height="824" alt="image" src="https://github.com/user-attachments/assets/ca61d1ba-e3de-461e-8fba-0f2065b17e29" />
<img width="524" height="628" alt="image" src="https://github.com/user-attachments/assets/41adefc7-0210-4100-bad6-8d32b369aba1" />
<img width="568" height="422" alt="image" src="https://github.com/user-attachments/assets/69fa2853-bb7e-4397-a700-2d16abccbe30" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
