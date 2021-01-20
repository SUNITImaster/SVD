# -*- coding: utf-8 -*-
"""
This code computes orthogonal Principal components from the given set of features
and compares it with sklearn.decomposition builtin fuction PCA

@author: Suniti
"""


import os
import pandas as pd
from sklearn import impute
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,LabelBinarizer,StandardScaler
from numpy.linalg import multi_dot
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg as la



os.chdir("C:/Users/Admin/Python")
train1=pd.read_csv("./data/train_titanic.csv")
test1=pd.read_csv("./data/test_titanic.csv")


print(train1.dtypes)


#computing missing count from string columns
print(train1.isnull().sum())
print(test1.isnull().sum())



train1['agebucket']=pd.cut(train1['Age'],5)
train1['agebucket']=train1['agebucket'].cat.add_categories('missing')
train1.loc[np.isnan(train1['Age'])].agebucket="missing"
pd.crosstab(index=train1['agebucket'].fillna('missing'),columns=train1["Survived"],dropna=False,normalize='index')

#missing value impuation
imp=impute.SimpleImputer(missing_values=np.nan,strategy="constant",fill_value=24)
train1['Age']=imp.fit_transform(train1['Age'].values.reshape(-1,1))
test1['Age']=imp.transform(test1['Age'].values.reshape(-1,1))
imp1=impute.SimpleImputer(missing_values=np.nan,strategy="most_frequent")
train1['Embarked']=imp1.fit_transform(train1['Embarked'].values.reshape(-1,1))
#dropping string columns like name and ticket no. as those are not required
train1.drop(columns=['Name','Ticket','Cabin','agebucket'],inplace=True)
test1.drop(columns=['Name','Ticket','Cabin'],inplace=True)
test1.dropna(inplace=True)

#One hot encoding of Object or String variables
oh=OneHotEncoder()
le=LabelEncoder()
lb=LabelBinarizer()
train1['Embarked_num']=le.fit_transform(train1['Embarked'])
test1['Embarked_num']=le.transform(test1['Embarked'])
tr=pd.DataFrame(oh.fit_transform(train1['Embarked_num'].values.reshape(-1,1)).toarray(),columns=["Embark_C","Embark_Q","Embark_S"])
ts=pd.DataFrame(oh.transform(test1['Embarked_num'].values.reshape(-1,1)).toarray(),columns=["Embark_C","Embark_Q","Embark_S"])

train1['Gender']=lb.fit_transform(train1['Sex'])
test1['Gender']=lb.fit_transform(test1['Sex'])
train2=pd.concat([train1,tr],axis=1)
test2=pd.concat([test1,ts],axis=1)

train2.drop(columns=['PassengerId','Sex','Embarked','Embarked_num'],inplace=True)
test2.drop(columns=['PassengerId','Sex','Embarked','Embarked_num'],inplace=True)
test2.dropna(inplace=True)

Y=train2.pop('Survived')

#we will now attempt to construct principal components on our own using SVD.


#normalizing the variables to variance of 1 
stdscaler=StandardScaler()
X_train_scaled = stdscaler.fit_transform(train2)

#compute  covariance matrix 
CovM=np.cov(X_train_scaled.T)

"""
compute eigenvalue and eigenvector decomposition of cov matrix using
scipy linalg module. The eigvec1 matrix below gives V matrix with  variable loadings 
on the factors. X*V eventually gives the desired orthogonal principal components
"""
eigval,eigvec=la.eig(CovM)
eigval=eigval.real
idx=np.argsort(eigval)[::-1]
eigval=eigval[idx]
eigvec1=eigvec[:,idx]
V=eigvec1 #This is V matrix is U*S*V.T
pcomp_our=multi_dot([X_train_scaled,V])


#now we will compute PCA using sklearn function and compare results

PCA_extractor=PCA(n_components=len(eigval))
pcomp_sklearn=PCA_extractor.fit_transform(X_train_scaled)

diff=np.sum(abs(np.square(pcomp_sklearn)-np.square(pcomp_our))).astype("int32")
print("The sum of absolute difference between the PCA squared components is "+np.str(diff))



