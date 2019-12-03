import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

titanic=pd.read_csv('0000000000002429_training_titanic_x_y_train.csv')

X=titanic.iloc[:,:-1]
Y=titanic.iloc[:,10]

X.isnull().sum()

X.Age.fillna(X.Age.mean(),inplace=True)
X.Cabin.fillna('U',inplace=True)

def f(s):
    return s[0]

X['cabin']=X.Cabin.apply(f)
X['Num_family']=X['Parch']+X['SibSp']

def fun(s):
    return ord(s)-65

X['cabin']=X.cabin.apply(fun)

X.drop('Name',axis=1,inplace=True)
X.drop('Cabin',axis=1,inplace=True)

def sex_fun(s):
    if(s=='male'):
        return 0
    else:
        return 1
    
X['Sex']=X.Sex.apply(sex_fun)

X.drop('Parch',axis=1,inplace=True)
X.drop('SibSp',axis=1,inplace=True)

def embarked_fun(s):
    if(s=='S'):
        return 0
    elif(s=='C'):
        return 1
    else:
        return 2

X['Embarked']=X.Embarked.apply(embarked_fun)

X.drop('Fare',axis=1,inplace=True)
X.drop('Ticket',axis=1,inplace=True)

from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y)

from sklearn.linear_model import LogisticRegression
clg=LogisticRegression()

clg.fit(X_train,Y_train)

clg.score(X_train,Y_train)

Y_pred=clg.predict(X_test)

dummy_wrong=Y_pred-Y_test

clg.score(X_test,Y_test)