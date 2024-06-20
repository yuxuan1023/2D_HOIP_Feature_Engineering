import os
import pandas as pd
import numpy as np
import ase
from ase.io import read, write
import numpy, math, random
#from visualise import view
from ase import Atoms
import sys
#from dscribe.descriptors import SOAP
import itertools
import matplotlib.pyplot as plt
#import tqdm
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from copy import deepcopy
from sklearn.linear_model import LogisticRegression



def get_train_test_dfs(df_data, train_size=0.9, test_size=0.1, shuffle_data=True,random_state=42):
    #if shuffle_data==True: shuffle(df_data, )
    print('--------------------train size: {}--------------------'.format(train_size))
    df_train, df_test = train_test_split(df_data,train_size=train_size, test_size=test_size, random_state=random_state, shuffle= shuffle_data)
    return df_train, df_test


df_modulus=pd.read_csv('df_modulus_154L.csv')
df_features=pd.read_csv('features_154Lv2.csv')
df_data=df_features.merge(df_modulus,on='idx', how='inner')
df_data.set_index('idx',inplace=True)
df_all_clean=df_data
dum_group=pd.get_dummies(df_all_clean['crystal'])
dum_group2=pd.get_dummies(df_all_clean['X'])
dum_group3=pd.get_dummies(df_all_clean['LYTTP'])
dum_group4=pd.get_dummies(df_all_clean['struc_type'])
df_all_clean=df_all_clean.join(dum_group)
df_all_clean=df_all_clean.join(dum_group2)
df_all_clean=df_all_clean.join(dum_group3)
df_all_clean=df_all_clean.join(dum_group4)
df_all_clean.dropna( inplace=True )
df_all_clean_new = df_all_clean.drop(['struc_type','crystal','X','LYTTP','E_out','E_in'],axis = 1)
#df_all_clean_new = df_all_clean_new.drop(df_all_clean_new.columns[1],axis = 1)
softline=24.75 #smaller than line if considered as soft
df_all_clean.loc[:,'E_out_class']=0
#df_all_clean.loc[df_all_clean.E_out>softline, 'E_out_class']=0
df_all_clean.loc[df_all_clean.E_out<softline, 'E_out_class']=1
#df_all_clean_new.pop('E_out')
#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/max(df_all_clean['E_out']) )
df_all_clean_new.loc[:, 'E_out_class']=df_all_clean['E_out_class']



from sklearn.model_selection import StratifiedShuffleSplit


# In[11]:


desc='E_out_class'
#rs=4139
#rs=4827
rs=839
#rs=217
X=df_all_clean_new.loc[:, df_all_clean_new.columns != desc]
y=df_all_clean_new[desc]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
for train_index, test_index in split.split(X, y):
    x_train = X.iloc[train_index].to_numpy()
    x_test = X.iloc[test_index].to_numpy()
    y_train = y.iloc[train_index].to_numpy()
    y_test = y.iloc[test_index].to_numpy()
print('**************{}******{}****************'.format(rs, len(x_train)) )
clf = LogisticRegression(solver='liblinear',max_iter=500, random_state=rs)
param_grid = {"C":    [.001, .01, .1, 1],
                  'class_weight': [None, 'balanced'],
              "penalty": ['l1','l2'],
              }
    
search = GridSearchCV(clf, param_grid, cv=5,verbose=10,                          scoring='roc_auc',return_train_score=True).fit(x_train, y_train)
print("The best hyperparameters are ",search.best_params_)
clf=LogisticRegression(C = search.best_params_["C"],
                           class_weight  = search.best_params_["class_weight"],
                           penalty = search.best_params_["penalty"],
                           solver='liblinear',max_iter=5000, random_state=rs)

clf.fit(x_train, y_train)
    #x_test=df_test.loc[:, df_test.columns != desc].to_numpy()
    #y_test=df_test[desc].to_numpy()
auc=accuracy_score(y_test,clf.predict(x_test))
roc=roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])


# In[12]:


auc,roc




