import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset=pd.read_csv('breast-cancer-wisconsin.data.txt')
dataset=dataset.replace('?',np.NaN)

dataset=dataset.drop(['id'],axis=1)
X=dataset.iloc[:,:9].values
y=dataset.iloc[:,9].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean')
X[:,:9]=imputer.fit_transform(X[:,:9])

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier_logistic=LogisticRegression(random_state=0)
classifier_logistic.fit(X_train,y_train)

y_pred_logistic=classifier_logistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cmd_logistic=confusion_matrix(y_test,y_pred_logistic)

from sklearn.neighbors import KNeighborsClassifier
classifier_Knn=KNeighborsClassifier(n_neighbors=5)
classifier_Knn.fit(X_train,y_train)

y_pred_knn=classifier_Knn.predict(X_test)

cmd_Knn=confusion_matrix(y_test,y_pred_knn)

from sklearn.svm import SVC
classifier_svm=SVC(kernel='poly')
classifier_k_svm=SVC(kernel='rbf')
classifier_svm.fit(X_train,y_train)
classifier_k_svm.fit(X_train,y_train)

y_pred_svm=classifier_svm.predict(X_test)
y_pred_k_svm=classifier_k_svm.predict(X_test)

cmd_svm=confusion_matrix(y_test,y_pred_svm)
cmd_k_svm=confusion_matrix(y_test,y_pred_k_svm)

from sklearn.naive_bayes import GaussianNB
classifier_naive_bayes=GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)

y_pred_naive_bayes=classifier_naive_bayes.predict(X_test)

cmd_naive_bayes=confusion_matrix(y_test,y_pred_naive_bayes)

from sklearn.tree import DecisionTreeClassifier
classifier_decision=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier_decision.fit(X_train,y_train)

y_pred_decision=classifier_decision.predict(X_test)

cmd_decision=confusion_matrix(y_test,y_pred_decision)

from sklearn.ensemble import RandomForestClassifier
classifier_random=RandomForestClassifier(n_estimators=5,random_state=0)
classifier_random.fit(X_train,y_train)

y_pred_random=classifier_random.predict(X_test)

cmd=confusion_matrix(y_test,y_pred_random)        