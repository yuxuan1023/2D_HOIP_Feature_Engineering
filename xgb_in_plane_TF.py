#!/usr/bin/env python
# coding: utf-8

# In[30]:


#this version uses the model picked from the best model after 200 randomly splitting
#the target E-out is saceled with the max values from the train data
#all features are calculated by average over all Pbs


# In[31]:


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


# In[32]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# In[33]:


from copy import deepcopy


# In[34]:


def get_train_test_dfs(df_data, train_size=0.9, test_size=0.1, shuffle_data=True,random_state=42):
    #if shuffle_data==True: shuffle(df_data, )
    print('--------------------train size: {}--------------------'.format(train_size))
    df_train, df_test = train_test_split(df_data,train_size=train_size, test_size=test_size, random_state=random_state, shuffle= shuffle_data)
    return df_train, df_test


# In[35]:


df_modulus=pd.read_csv('df_modulus_30L.csv')


# In[36]:


df_modulus


# In[37]:


df_features=pd.read_csv('features_30L.csv')


# In[38]:


df_features


# In[39]:


df_data=df_features.merge(df_modulus,on='idx', how='inner')


# In[40]:


len(df_modulus), len(df_features), len(df_data)


# In[41]:


#df_data.loc[df_data.idx==26]


# In[42]:


df_data.set_index('idx',inplace=True)


# In[43]:


df_all_clean=df_data


# In[44]:


dum_group=pd.get_dummies(df_all_clean['crystal'])


# In[45]:


dum_group2=pd.get_dummies(df_all_clean['X'])
dum_group3=pd.get_dummies(df_all_clean['LYTTP'])


# In[46]:


df_all_clean=df_all_clean.join(dum_group)
df_all_clean.loc[:,'triclinic']=0
df_all_clean=df_all_clean.join(dum_group2)
df_all_clean=df_all_clean.join(dum_group3)
df_all_clean.loc[:,'triclinic-cell']=0
df_all_clean.loc[:,'100']=1
df_all_clean.loc[:,'110']=0


# In[47]:


df_all_clean


# In[48]:


#df_all_clean.set_index('idx',inplace=True)


# In[49]:


df_all_clean.dropna( inplace=True )


# In[50]:


df_all_clean_new = df_all_clean.drop(['crystal','X','LYTTP','E_out','E_in'],axis = 1)
#df_all_clean_new = df_all_clean_new.drop(df_all_clean_new.columns[1],axis = 1)


# In[51]:


#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/max(df_all_clean['E_out']) )
#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/35.09091569 )


# In[52]:


softline=16.30 #smaller than line if considered as soft
df_all_clean.loc[:,'E_in_class']=0
#df_all_clean.loc[df_all_clean.E_out>softline, 'E_out_class']=0
df_all_clean.loc[df_all_clean.E_in<softline, 'E_in_class']=1
df_all_clean_new.loc[:, 'E_in_class']=df_all_clean['E_in_class']


# In[53]:


df_all_clean_new.columns


# In[54]:


desc='E_in_class'
df_test=df_all_clean_new
#x_train=df_train.loc[:, df_train.columns != desc].to_numpy()
#y_train=df_train[desc].to_numpy()
x_test=df_test.loc[:, df_test.columns != desc].to_numpy()
y_test=df_test[desc].to_numpy()


# In[55]:


df_all_clean[['E_in', 'E_in_class']]


# In[56]:


#xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# In[57]:


clf=xgb.XGBClassifier(verbosity = 0, random_state=42)clf.load_model("model_in_TF.json")
y_pred = clf.predict(x_test)

print('y_pred',y_pred)
print('y_test',y_test)
print( accuracy_score(y_test,y_pred), roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))


