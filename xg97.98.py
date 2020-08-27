# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:11:23 2020

@author: theod
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:21:12 2020

@author: theod
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:31:26 2020

@author: theod
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#from tensorflow import random
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
sns.set() # setting seaborn default for plots
import scipy.stats as ss


#  Inserting the Training dataset
train  = pd.read_csv('train.csv')
x_train = pd.DataFrame(train.iloc[: , :-1].values, columns = ['state',	'account_length',	'area_code',	'international_plan',	'voice_mail_plan',	'number_vmail_messages',	'total_day_minutes',	'total_day_calls',	'total_day_charge',	'total_eve_minutes',	'total_eve_calls',	'total_eve_charge',	'total_night_minutes',	'total_night_calls',	'total_night_charge',	'total_intl_minutes',	'total_intl_calls',	'total_intl_charge',	'number_customer_service_calls'])

#,'voice_mail_plan'
plt.figure(figsize=(15,8))
sns.heatmap(train.drop(['state','area_code','international_plan','voice_mail_plan'],axis=1).corr(), vmax=0.6, square=True, annot=True)

x_train['total minutes'] = x_train['total_day_minutes'] + x_train['total_eve_minutes'] + x_train['total_night_minutes'] + x_train['total_intl_minutes']
x_train['total charge'] = x_train['total_day_charge'] + x_train['total_eve_charge'] + x_train['total_night_charge']+ x_train['total_intl_charge']
x_train['total calls'] = x_train['total_day_calls'] + x_train['total_eve_calls'] + x_train['total_night_calls']+ x_train['total_intl_calls']


x_train['total minutes1'] = x_train['total_day_minutes'] +  x_train['total_intl_minutes']
x_train['total charge1'] = x_train['total_day_charge'] + x_train['total_intl_charge']
x_train['total calls1'] = x_train['total_day_calls'] + x_train['total_intl_calls']


x_train.drop(['total_day_minutes','total_eve_minutes','total_night_minutes','total_intl_minutes','account_length',	'area_code'], axis=1, inplace=True)

train.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10))
pd.crosstab(train['state'], train['churn'])

x_train.drop(['state'], axis=1, inplace=True)

#x_train = pd.get_dummies(x_train, columns=['area_code'], prefix = 'area_code')
#x_train.drop('area_code_area_code_510', axis=1, inplace=True)
x_train = pd.get_dummies(x_train, columns=['international_plan'], prefix = 'international_plan')
x_train.drop('international_plan_yes', axis=1, inplace=True)
x_train = pd.get_dummies(x_train, columns=['voice_mail_plan'], prefix = 'voice_mail_plan')
x_train.drop('voice_mail_plan_yes', axis=1, inplace=True)

fil = (x_train['total_day_calls']<x_train['total_day_calls'].mean()) & (x_train['total_day_charge']> x_train['total_day_charge'].mean())
x_train.loc[fil].head()
df_temp = x_train[fil]
df_temp.shape


y_train = pd.DataFrame(train.iloc[:, -1].values, columns = ['churn'])
#y_train['churn'] = y_train['churn'].apply(lambda row: 1 if row=='yes' else 0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

print(x_train)
#  Inserting the Test dataset 
x_test = pd.read_csv('test.csv')
x_test = pd.DataFrame(x_test.iloc[: , 1:].values, columns = ['state',	'account_length',	'area_code',	'international_plan',	'voice_mail_plan',	'number_vmail_messages',	'total_day_minutes',	'total_day_calls',	'total_day_charge',	'total_eve_minutes',	'total_eve_calls',	'total_eve_charge',	'total_night_minutes',	'total_night_calls',	'total_night_charge',	'total_intl_minutes',	'total_intl_calls',	'total_intl_charge',	'number_customer_service_calls'])

x_test['total minutes'] = x_test['total_day_minutes'] + x_test['total_eve_minutes'] + x_test['total_night_minutes']+ x_test['total_intl_minutes']
x_test['total charge'] = x_test['total_day_charge'] + x_test['total_eve_charge'] + x_test['total_night_charge']+ x_test['total_intl_charge']
x_test['total calls'] = x_test['total_day_calls'] + x_test['total_eve_calls'] + x_test['total_night_calls']+ x_test['total_intl_calls']


x_test['total minutes1'] = x_test['total_day_minutes'] + x_test['total_intl_minutes']
x_test['total charge1'] = x_test['total_day_charge'] + x_test['total_intl_charge']
x_test['total calls1'] = x_test['total_day_calls'] + x_test['total_intl_calls']


x_test.drop(['total_day_minutes','total_eve_minutes','total_night_minutes','total_intl_minutes','account_length',	'area_code'], axis=1, inplace=True)

x_test.drop('state', axis=1, inplace=True)
#x_test = pd.get_dummies(x_test, columns=['area_code'], prefix = 'area_code')
#x_test.drop('area_code_area_code_510', axis=1, inplace=True)
x_test = pd.get_dummies(x_test, columns=['international_plan'], prefix = 'international_plan')
x_test.drop('international_plan_yes', axis=1, inplace=True)
x_test = pd.get_dummies(x_test, columns=['voice_mail_plan'], prefix = 'voice_mail_plan')
x_test.drop('voice_mail_plan_yes', axis=1, inplace=True)

y_test = pd.read_csv('sampleSubmission.csv')
y_test = pd.DataFrame(y_test.iloc[: , 1:].values, columns = ['churn']) 
y_test['churn'] = y_test['churn'].apply(lambda row: 1 if row=='yes' else 0)




sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate =  0.04, max_depth = 10, n_estimators = 250,early_stopping_rounds = 10,  reg_lambda = 1,colsample_bytree = 0.8, gamma = 0,min_child_weight= 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from sklearn.model_selection import GridSearchCV 


#, gamma = 0.1, min_child_weight = 2 ,, colsample_bytree = 0.9
#'min_child_weight':[1, 2], 'gamma':[0, 0.05, 0.1], 'colsample_bytree':[0.8, 0.9, 1.0]
# defining parameter range 
param_grid = {'max_depth': [  10],  
              'learning_rate': [ 0.03, 0.02, 0.04],
              'early_stopping_rounds' :[ 10],
              'reg_lambda' : [1, 0.1, 0.01],
              'n_estimators': [ 200, 250],
              'min_child_weight':[1, 2], 'gamma':[0, 0.05, 0.1], 'colsample_bytree':[0.8, 0.9, 1.0]
              }  
  
grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(x_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

im = classifier.feature_importances_

