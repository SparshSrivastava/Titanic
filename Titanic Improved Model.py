import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

titanic=pd.read_csv('train.csv')

X=titanic.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]]
Y=titanic.iloc[:,1]

X.isnull().sum()

X.Age.fillna(X.Age.mean(),inplace=True)
X.Cabin.fillna('U',inplace=True)
X.Embarked.fillna('U',inplace=True)

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
    elif(s=='Q'):
        return 2
    else:
        return 3

X['Embarked']=X.Embarked.apply(embarked_fun)

X.drop('Fare',axis=1,inplace=True)
X.drop('Ticket',axis=1,inplace=True)

from sklearn.linear_model import LogisticRegression
clg=LogisticRegression()

clg.fit(X,Y)

clg.score(X,Y)

titanic_test=pd.read_csv('test.csv')

X_test=titanic_test.iloc[:,:]
X_test.isnull().sum()

X_test.Age.fillna(X.Age.mean(),inplace=True)
X_test.Cabin.fillna('U',inplace=True)
X_test.Embarked.fillna('U',inplace=True)

X_test['cabin']=X_test.Cabin.apply(f)
X_test['Num_family']=X_test['Parch']+X_test['SibSp']

X_test['cabin']=X_test.cabin.apply(fun)

X_test.drop('Name',axis=1,inplace=True)
X_test.drop('Cabin',axis=1,inplace=True)
    
X_test['Sex']=X_test.Sex.apply(sex_fun)

X_test.drop('Parch',axis=1,inplace=True)
X_test.drop('SibSp',axis=1,inplace=True)

X_test['Embarked']=X_test.Embarked.apply(embarked_fun)

X_test.drop('Fare',axis=1,inplace=True)
X_test.drop('Ticket',axis=1,inplace=True)

Y_pred=clg.predict(X_test)

ans=titanic_test.iloc[:,0]
ans['Survived']=Y_pred

from sklearn.linear_model import LogisticRegression
clg2=LogisticRegression(solver='saga',multi_class='multinomial',max_iter=1500)

clg2.fit(X,Y)

Y_pred_Saga=clg.predict(X_test)

print(Y_pred_Saga-Y_pred)

clg3=LogisticRegression(solver='saga',max_iter=2000)
clg3.fit(X,Y)

Y_pred_Saga_2=clg3.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_pred,Y_pred_Saga_2)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=6)

rf.fit(X,Y)

Y_pred_rf=rf.predict(X_test)