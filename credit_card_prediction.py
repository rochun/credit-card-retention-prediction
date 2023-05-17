# -*- coding: utf-8 -*-
"""
### **Links:**

Data: https://www.kaggle.com/sakshigoyal7/credit-card-customers

Blog: https://mfiyjuz.medium.com/bank-customer-retention-c6a89058d358

Github: https://github.com/roycechun/bank-customer-retention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import linear_model, datasets

cc_df = pd.read_csv('Desktop/creditcard.csv')
cc_df.head()

cc_df.info()

cc_df.shape

cc_df.dtypes

cc_df = cc_df.drop(columns=['CLIENTNUM','Contacts_Count_12_mon','Unnamed: 21'])
cc_df.head()

cc_df.isna().sum()

# Credit Limit density chart to determine the distribution
sns.histplot(x=cc_df['Credit_Limit'], stat='density' ,bins=80,kde=True)
plt.title('Credit Limit Density Chart')
plt.xlabel('Credit Limit')
plt.ylabel('Density')
plt.show()

# From the histogram, we can see that there is an overwhelming amount of 0.0 utilization rates in the dataset
sns.histplot(x=cc_df['Avg_Utilization_Ratio'] ,bins=80,kde=True)
plt.title('Average Utilization Ratio Density')
plt.xlabel('Average Utilization Ratio')
plt.ylabel('Count')
plt.show()

# Determine whether there is a normal distribution of transaction counts
sns.histplot(x=cc_df['Total_Trans_Ct'], stat='density' ,bins=80,kde=True)
plt.title('Total Transactions Count Density')
plt.xlabel('Total Transactions Count')
plt.ylabel('Density')
plt.show()

# plot shows an unbalance of the target values
sns.countplot(x=cc_df['Attrition_Flag'])
plt.show()

X = cc_df.iloc[:, 1:]
y = cc_df.iloc[:,0]

print(cc_df.shape)
print(X.shape)
print(y.shape)

# get dummy variables
X_dummy = pd.get_dummies(X, columns=X.columns)

# split dataset
xTrain, xTest, yTrain, yTest = train_test_split(X_dummy, y, test_size = 0.3, random_state = 42, stratify=y)

# Baseline Accuracy aka Dummy Classifier
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(xTrain,yTrain)
base_accuracy = dummy_classifier.score(xTest,yTest)
print('Baseline Accuracy:', base_accuracy)

# Logistic Regression

logreg = LogisticRegression(max_iter=10000)

logreg.fit(xTrain,yTrain)

logreg_pred = logreg.predict(xTest)

logreg_acc = accuracy_score(yTest, logreg_pred)

print("Logistic Regression Accuracy:", logreg_acc)

"""### Bagging

"""

# Bagging classifier 

bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=42)

bagging.fit(xTrain, yTrain)

bagging_pred = bagging.predict(xTest)

bagging_acc = accuracy_score(yTest, bagging_pred)

print("Bagging Accuracy:", bagging_acc)

# Random forest classifier with n_estimators = 100, max_features = 7, and random_state = 42

rf = RandomForestClassifier(n_estimators=100, max_features=7, random_state=42)

rf.fit(xTrain, yTrain)

rf_pred = rf.predict(xTest)

rf_acc = accuracy_score(yTest, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# Ada Boosting Classifier with n_estimators = 200, random_state = 42, and learning_rate = 0.05

base_est = DecisionTreeClassifier(max_depth=5)

adaboost = AdaBoostClassifier(base_est, n_estimators=200, random_state=42, learning_rate=0.05)

adaboost.fit(xTrain, yTrain)

adaboost_pred = adaboost.predict(xTest)

adaboost_acc = accuracy_score(yTest, adaboost_pred)



print("Ada Boosting Accuracy:", adaboost_acc)

# Gradient Boosting Classifier with random_state = 42

gradboost = GradientBoostingClassifier(random_state=42)


gradboost.fit(xTrain, yTrain)

gradboost_pred = gradboost.predict(xTest)

gradboost_acc = accuracy_score(yTest, gradboost_pred)

print("Gradient Boosting Accuracy:", gradboost_acc)

# Voting Classifier with random_state = 42 & max_iter = 100000

logClf = LogisticRegression(random_state=42, max_iter=10000)

svmClf = SVC(probability=True)

rfClf = RandomForestClassifier(n_estimators=100, max_features=7, random_state=42)

voting = VotingClassifier(estimators=[('log',logClf), ('svm', svmClf),('rf', rfClf)], voting='soft', verbose=True, n_jobs=-1)

voting.fit(xTrain, yTrain)

voting_pred = voting.predict(xTest)

voting_acc = accuracy_score(yTest, voting_pred)

print('Voting Accuracy:', voting_acc)

# set parameters for the randomized search cross validation algorithm

logregv2 = LogisticRegression(max_iter=10000, random_state=42)

p = ['l1','l2']

# uniform distribution between C-values
C = uniform(loc=0, scale=4)

param = dict(C=C, penalty=p)

logreg_tuned = RandomizedSearchCV(logregv2, param_distributions=param, random_state=42, cv=3, n_iter=20, verbose=2, n_jobs=-1)

logreg_tuned.fit(xTrain,yTrain)

logreg_tuned.best_params_

# Test model between the testing and training sets

fine_tuned_logreg = LogisticRegression(C=3.33,penalty='l2', max_iter=10000)

fine_tuned_logreg.fit(xTrain, yTrain)

# testing set
fine_test_pred = fine_tuned_logreg.predict(xTest)
test_accuracy = accuracy_score(fine_test_pred, yTest)

# training set
fine_train_pred = fine_tuned_logreg.predict(xTrain)
train_accuracy = accuracy_score(fine_train_pred, yTrain)


cf = metrics.confusion_matrix(yTest, fine_test_pred)

cf2 = metrics.confusion_matrix(yTrain, fine_train_pred)

print(cf)
print(cf2)
print('')
print('Accuracy on training partition:', train_accuracy)
print('Accuracy on testing partition:', test_accuracy)

# Show the differences between the models in visual form

x_val = ['Baseline', 'Logistic', 'Bagging', 'RF', 'AdaBoost', 'GradBoost','Voting','FineTuned']


# hold score data in list
acc_score = [base_accuracy,logreg_acc, bagging_acc, rf_acc, adaboost_acc, gradboost_acc, voting_acc, test_accuracy]

sns.barplot(x=x_val, y=acc_score)
plt.ylim(0.8,0.95)
plt.tight_layout()

plt.title('Difference Between Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.show()
