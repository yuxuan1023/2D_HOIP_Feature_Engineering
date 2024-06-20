import os
import pandas as pd
import numpy as np
import ase
from ase.io import read, write
import numpy, math, random
import pickle
#from visualise import view
from ase import Atoms
import sys
#from dscribe.descriptors import SOAP
import itertools
import matplotlib.pyplot as plt
#import tqdm
from tqdm import tqdm
import seaborn as sns
from copy import deepcopy
from sklearn import preprocessing
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.interpolate import interp1d, pchip


import xgboost as xgb
#import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix,classification_report,    roc_auc_score,accuracy_score
from sklearn.utils import shuffle

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# In[4]:


import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
#from __future__ import print_function


# In[5]:


from lime import lime_text
from sklearn.pipeline import make_pipeline


# In[6]:


def get_train_test_dfs(df_data, train_size=0.9, test_size=0.1, shuffle_data=True,random_state=42):
    #if shuffle_data==True: shuffle(df_data, )
    print('--------------------train size: {}--------------------'.format(train_size))
    df_train, df_test = train_test_split(df_data,train_size=train_size, test_size=test_size, random_state=random_state, shuffle= shuffle_data)
    return df_train, df_test


# In[7]:


df_modulus=pd.read_csv('df_modulus_154L.csv')


# In[8]:


df_modulus


# In[9]:


df_features=pd.read_csv('features_154Lv2.csv')


# In[10]:


df_data=df_features.merge(df_modulus,on='idx', how='inner')


# In[11]:


df_data.set_index('idx',inplace=True)


# In[12]:


df_all_clean=df_data


# In[13]:


dum_group=pd.get_dummies(df_all_clean['crystal'])


# In[14]:


dum_group2=pd.get_dummies(df_all_clean['X'])
dum_group3=pd.get_dummies(df_all_clean['LYTTP'])
dum_group4=pd.get_dummies(df_all_clean['struc_type'])


# In[15]:


df_all_clean=df_all_clean.join(dum_group)


# In[16]:


df_all_clean=df_all_clean.join(dum_group2)
df_all_clean=df_all_clean.join(dum_group3)
df_all_clean=df_all_clean.join(dum_group4)


# In[17]:


#df_all_clean.set_index('idx',inplace=True)


# In[18]:


df_all_clean.dropna( inplace=True )


# In[19]:


df_all_clean_new = df_all_clean.drop(['struc_type','crystal','X','LYTTP','E_out','E_in'],axis = 1)
#df_all_clean_new = df_all_clean_new.drop(df_all_clean_new.columns[1],axis = 1)


# In[20]:


softline=16.30 #smaller than line if considered as soft
df_all_clean.loc[:,'E_in_class']=0
#df_all_clean.loc[df_all_clean.E_out>softline, 'E_out_class']=0
df_all_clean.loc[df_all_clean.E_in<softline, 'E_in_class']=1


# In[21]:


df_all_clean[['E_in', 'E_in_class']]


# In[22]:


df_all_clean['E_in'].describe()


# In[23]:


#df_all_clean_new.pop('E_out')
#df_all_clean_new.insert(len(df_all_clean_new.columns), 'E_out', \
#                        df_all_clean['E_out']/max(df_all_clean['E_out']) )
df_all_clean_new.loc[:, 'E_in_class']=df_all_clean['E_in_class']


# In[24]:


desc='E_in_class'
rs=4827
df_train, df_test=get_train_test_dfs(df_all_clean_new, train_size=0.8, test_size=0.2, random_state=rs)
print('*********************{}*************************'.format(len(df_train)))
x_train=df_train.loc[:, df_train.columns != desc].to_numpy()
y_train=df_train[desc].to_numpy()
x_test=df_test.loc[:, df_test.columns != desc].to_numpy()
y_test=df_test[desc].to_numpy()


errors = classifier.predict_proba(x_test)[:,1] - y_test
sorted_errors = np.argsort(abs(errors))
worse_5 = sorted_errors[-5:]
best_5 = sorted_errors[:5]

#print(pd.DataFrame({'worse':errors[worse_5]}))
#print()
#print(pd.DataFrame({'best':errors[best_5]}))


# In[28]:


best_10 = sorted_errors[:10]
best_20 = sorted_errors[:20]


# In[29]:


feature_names=df_train.columns[:-1].to_list()


# In[30]:


len(feature_names)


# In[31]:


sns.color_palette(palette='Blues_r')


# In[32]:


sns.color_palette(palette='Purples_r')


# In[33]:


colors=sns.color_palette(palette='Blues_r')


# In[34]:


xgb.plot_importance(classifier,max_num_features=10,importance_type = "weight")


# In[35]:


# the mean and std, and discretize it into quartiles


# In[36]:


def numerial_feature_trend(df_train, x_test, classifier, quantiles_all, feature_name, data_point):
    feature_names=df_train.columns[:-1].to_list()
    feature_idx = feature_names.index(feature_name)
    temp = x_test[data_point].copy()
    #print(temp[feature_idx],'P(soft) before:', classifier.predict_proba(temp.reshape(1,-1))[0,1])
    probs=[]
    for i, q in enumerate(quantiles_all):
        temp[feature_idx] = q
        #print(q,'P(soft) {}%:'.format(i*0.25*100), classifier.predict_proba(temp.reshape(1,-1))[0,1])      
        probs.append(classifier.predict_proba(temp.reshape(1,-1))[0,1])
    return probs


# In[37]:


#features_4=['Pb_X_Pb', 'bond_angle_variance', 'N_penetration', 'ax_bond_length']
features_4=['Pb_X_Pb', 'ax_bond_length', 'out_of_plane_disortion','bond_angle_variance']
todo_idx=[best_10[0],best_10[2],best_10[4]]


# In[38]:


feature_name=features_4[0]
feature_idx = feature_names.index(feature_name)
boxplot=df_train.boxplot(column=feature_name,return_type='dict')
quantiles_all=[]
caps = boxplot['caps']
capbottom = caps[0].get_ydata()[0]
captop = caps[1].get_ydata()[0]
quantiles_all.append(capbottom)
quantiles_all.extend(list(df_train.iloc[:,feature_idx].quantile([0.25, 0.5, 0.75])))
quantiles_all.append(captop)
quantiles_all


# In[39]:


probs_all1=[]
#todo_idx=[best_10[0],best_10[1],best_10[2]]
for i in todo_idx:
    probs=numerial_feature_trend(df_train, x_test, classifier, np.linspace(min(df_train[feature_name].to_list()),                     max(df_train[feature_name].to_list()), 20), 'Pb_X_Pb', i)
    probs_all1.append(probs)
    print('*****************************')


# In[40]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all1[0])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.95)])



feature_name=features_4[1]
#feature_name='ax_bond_length'
feature_idx = feature_names.index(feature_name)
boxplot=df_train.boxplot(column=feature_name,return_type='dict')
quantiles_all=[]
caps = boxplot['caps']
capbottom = caps[0].get_ydata()[0]
captop = caps[1].get_ydata()[0]
quantiles_all.append(capbottom)
quantiles_all.extend(list(df_train.iloc[:,feature_idx].quantile([0.25, 0.5, 0.75])))
quantiles_all.append(captop)
quantiles_all



probs_all2=[]
#for i in best_5:
#todo_idx=[best_20[0], best_20[1], best_20[2] ]
for i in todo_idx:
    probs=numerial_feature_trend(df_train, x_test, classifier, np.linspace(min(df_train[feature_name].to_list()),                                                             max(df_train[feature_name].to_list()), 20), feature_name, i)
    probs_all2.append(probs)
    print('*****************************')


# In[46]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all2[1])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.95)])


# In[47]:


feature_name=features_4[2]
feature_idx = feature_names.index(feature_name)
boxplot=df_train.boxplot(column=feature_name,return_type='dict',showfliers=False)
quantiles_all=[]
caps = boxplot['caps']
capbottom = caps[0].get_ydata()[0]
captop = caps[1].get_ydata()[0]
quantiles_all.append(capbottom)
quantiles_all.extend(list(df_train.iloc[:,feature_idx].quantile([0.25, 0.5, 0.75])))
quantiles_all.append(captop)
quantiles_all


# In[48]:


probs_all3=[]
#for i in best_5:
#todo_idx=[best_20[0], best_20[1], best_20[2] ]
for i in todo_idx:
    probs=numerial_feature_trend(df_train, x_test, classifier,         np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20 ), feature_name, i)
    probs_all3.append(probs)
    print('*****************************')


# In[49]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all3[1])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.95)])




feature_name=features_4[3]
feature_idx = feature_names.index(feature_name)
boxplot=df_train.boxplot(column=feature_name,return_type='dict')
quantiles_all=[]
caps = boxplot['caps']
capbottom = caps[0].get_ydata()[0]
captop = caps[1].get_ydata()[0]
quantiles_all.append(capbottom)
quantiles_all.extend(list(df_train.iloc[:,feature_idx].quantile([0.25, 0.5, 0.75])))
quantiles_all.append(captop)



probs_all4=[]
#for i in best_5:
#todo_idx=[best_20[0], best_20[1], best_20[2] ]
#for i in best_10:
for i in todo_idx:
    probs=numerial_feature_trend(df_train, x_test, classifier,             np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20 ), feature_name, i)
    probs_all4.append(probs)
    print('*****************************')


# In[54]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all4[0])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.95)])


# In[57]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all4[1])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.95)])


# In[58]:


x=np.linspace(min(df_train[feature_name].to_list()), max(df_train[feature_name].to_list()), 20)
y=preprocessing.minmax_scale(probs_all4[1])
x_smooth = np.linspace(x[0], x[-1], 100)
f = pchip(x, y)
#print(x_smooth)
#print(f(x_smooth))
print(x_smooth[np.where(f(x_smooth)>0.8)])



import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(                        x_train, 
                        feature_names=df_train.columns, 
                        class_names=['non-soft','soft'], 
                        verbose=True, 
                        mode='classification')


best_10 = sorted_errors[:10]
best_20 = sorted_errors[:20]

# In[60]:


i=best_20[0]
exp1 = explainer.explain_instance(x_test[i], classifier.predict_proba, num_features=6, top_labels=1)
#print('Error: {}'.format( abs(exp.local_pred-exp.predict_proba[1])/exp.predict_proba[1]   ))
exp1.show_in_notebook(show_table=True, show_all=False)


# In[61]:


exp2 = explainer.explain_instance(x_test[best_10[1]], classifier.predict_proba, num_features=6, top_labels=1)
#print('Error: {}'.format( abs(exp.local_pred-exp.predict_proba[1])/exp.predict_proba[1]   ))
exp2.show_in_notebook(show_table=True, show_all=False)


# In[62]:


exp3 = explainer.explain_instance(x_test[best_10[2]], classifier.predict_proba, num_features=6, top_labels=1)
#print('Error: {}'.format( abs(exp.local_pred-exp.predict_proba[1])/exp.predict_proba[1]   ))
exp3.show_in_notebook(show_table=True, show_all=False)


# In[63]:


exp4 = explainer.explain_instance(x_test[best_10[3]], classifier.predict_proba, num_features=6, top_labels=1)
#print('Error: {}'.format( abs(exp.local_pred-exp.predict_proba[1])/exp.predict_proba[1]   ))
exp4.show_in_notebook(show_table=True, show_all=False)


# In[64]:


exp5 = explainer.explain_instance(x_test[best_10[4]], classifier.predict_proba, num_features=6, top_labels=1)
#print('Error: {}'.format( abs(exp.local_pred-exp.predict_proba[1])/exp.predict_proba[1]   ))
exp5.show_in_notebook(show_table=True, show_all=False)


# In[65]:


exp6 = explainer.explain_instance(x_test[best_33[5]], classifier.predict_proba, num_features=8, top_labels=1)
#print('Error: {}'.format( abs(exp6.local_pred-exp6.predict_proba[1])/exp6.predict_proba[1]   ))
exp6.show_in_notebook(show_table=True)


# In[66]:


y_test[best_20[12]]


# In[67]:


df1=pd.DataFrame(exp1.as_list(),columns=['feature','score'])
mask = df1['score'] < 0
df1['neg'] = df1['score'].mask(mask)
df1['pos'] = df1['score'].mask(~mask)


# In[68]:


df2=pd.DataFrame(exp2.as_list(),columns=['feature','score'])
mask = df2['score'] < 0
df2['neg'] = df2['score'].mask(mask)
df2['pos'] = df2['score'].mask(~mask)


# In[69]:


df3=pd.DataFrame(exp3.as_list(),columns=['feature','score'])
mask = df3['score'] < 0
df3['neg'] = df3['score'].mask(mask)
df3['pos'] = df3['score'].mask(~mask)


# In[70]:


#df4=pd.DataFrame(exp4.as_list(),columns=['feature','score'])
#mask = df4['score'] < 0
#df4['neg'] = df4['score'].mask(mask)
#df4['pos'] = df4['score'].mask(~mask)


# In[71]:


df5=pd.DataFrame(exp5.as_list(),columns=['feature','score'])
mask = df5['score'] < 0
df5['neg'] = df5['score'].mask(mask)
df5['pos'] = df5['score'].mask(~mask)


# In[72]:


df6=pd.DataFrame(exp6.as_list(),columns=['feature','score'])
mask = df6['score'] < 0
df6['neg'] = df6['score'].mask(mask)
df6['pos'] = df6['score'].mask(~mask)


# In[73]:


df1.feature.to_list()


# In[74]:


df2.feature.to_list()


# In[75]:


df3.feature.to_list()


# In[76]:


#df4.feature.to_list()


# In[77]:


df5.feature.to_list()


# In[78]:


df6.feature.to_list()


