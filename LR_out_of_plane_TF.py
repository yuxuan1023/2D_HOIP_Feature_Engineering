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


# In[27]:


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


# In[28]:


from copy import deepcopy


# In[29]:


def get_train_test_dfs(df_data, train_size=0.9, test_size=0.1, shuffle_data=True,random_state=42):
    #if shuffle_data==True: shuffle(df_data, )
    print('--------------------train size: {}--------------------'.format(train_size))
    df_train, df_test = train_test_split(df_data,train_size=train_size, test_size=test_size, random_state=random_state, shuffle= shuffle_data)
    return df_train, df_test


# In[30]:


df_modulus=pd.read_csv('df_modulus_30L.csv')


# In[31]:


df_modulus


# In[32]:


df_features=pd.read_csv('features_30L.csv')


# In[33]:


df_features


# In[34]:


df_data=df_features.merge(df_modulus,on='idx', how='inner')


# In[35]:


len(df_modulus), len(df_features), len(df_data)


# In[36]:


#df_data.loc[df_data.idx==26]


# In[37]:


df_data.set_index('idx',inplace=True)


# In[38]:


df_all_clean=df_data


# In[40]:


dum_group=pd.get_dummies(df_all_clean['crystal'])


# In[41]:


dum_group2=pd.get_dummies(df_all_clean['X'])
dum_group3=pd.get_dummies(df_all_clean['LYTTP'])


# In[42]:


df_all_clean=df_all_clean.join(dum_group)
df_all_clean.loc[:,'triclinic']=0


# In[43]:


df_all_clean=df_all_clean.join(dum_group2)
df_all_clean=df_all_clean.join(dum_group3)
df_all_clean.loc[:,'triclinic-cell']=0
df_all_clean.loc[:,'100']=1
df_all_clean.loc[:,'110']=0


# In[44]:


df_all_clean


# In[45]:


#df_all_clean.set_index('idx',inplace=True)


# In[46]:


#df_all_clean.dropna( inplace=True )


# In[47]:


df_all_clean_new = df_all_clean.drop(['crystal','X','LYTTP','E_out','E_in'],axis = 1)
#df_all_clean_new = df_all_clean_new.drop(df_all_clean_new.columns[1],axis = 1)


# In[48]:


#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/max(df_all_clean['E_out']) )
#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/35.09091569 )


# In[49]:


softline=24.75 #smaller than line if considered as soft
df_all_clean.loc[:,'E_out_class']=0
#df_all_clean.loc[df_all_clean.E_out>softline, 'E_out_class']=0
df_all_clean.loc[df_all_clean.E_out<softline, 'E_out_class']=1
df_all_clean_new.loc[:, 'E_out_class']=df_all_clean['E_out_class']


# In[50]:


df_all_clean_new.columns


# In[51]:


desc='E_out_class'
df_test=df_all_clean_new
#x_train=df_train.loc[:, df_train.columns != desc].to_numpy()
#y_train=df_train[desc].to_numpy()
x_test=df_test.loc[:, df_test.columns != desc].to_numpy()
y_test=df_test[desc].to_numpy()


# In[52]:


df_all_clean[['E_out', 'E_out_class']]


# In[53]:


#xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pickle


# In[57]:


clf = pickle.load(open('model_out_lr_TF', 'rb'))
y_pred = clf.predict(x_test)

print('y_pred',y_pred)
print('y_test',y_test)



