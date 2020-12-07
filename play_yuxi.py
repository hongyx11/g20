#!/usr/bin/env python
# coding: utf-8


# download the dataset and save it in the same dir with this file.
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RationalQuadratic,Exponentiation
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import os
kf = KFold(n_splits=10)
# import scikit learn
from sklearn import linear_model,svm
from sklearn.preprocessing import PolynomialFeatures
import sys

if len(sys.argv) != 2:
  print("you need specify poly degree")

# In[3]:


def getxandy(filename):
  with open(filename,'r') as f:
    df = pd.read_csv(f,index_col=0)
    x = np.zeros((len(df.x),2))
    y = np.zeros((len(df.x)))
    x[:,0] = df.x.to_numpy()
    x[:,1] = df.y.to_numpy()
    y = df.z.to_numpy()
  return x,y

# In[10]:


def getpolyxandy(filename,poly=2,withy=True):
  with open(filename,'r') as f:
    df = pd.read_csv(f,index_col=0)
    x = np.zeros((len(df.x),2))
    y = np.zeros((len(df.x)))
    x[:,0] = df.x.to_numpy()
    x[:,1] = df.y.to_numpy()
    if withy:
      y = df.z.to_numpy()
    else:
      y = None
    poly = PolynomialFeatures(degree=poly)
    x = poly.fit_transform(x)
  return x,y

# In[5]:


def scoreperdataset(y_pred,y_test):
  return np.sqrt(np.sum(np.square(y_pred - y_test))/len(y_pred))

# In[6]:


def writeresults(X_test,y_test,y_pred,f):
    idx = np.argsort(X_test[:,0])
    X_test = X_test[idx]
    y_test = y_test[idx]
    y_pred = y_pred[idx]
    data = {}
    data['x'] = X_test[:,0]
    data['y'] = X_test[:,1]
    data['z_test'] = y_test[idx]
    data['z_pred'] = y_pred[idx]
    df = pd.DataFrame(data)
    if not os.path.isdir('results/'):
      os.mkdir('results')
    df.to_csv('results/'+f+'.csv',float_format="%.6f")



# In[20]:


print("start to work on small datasets")
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
final_acc = []
pred_array = {}
trainfolder = 'small_datasets/'
datasets = [chr(ord('A') + x) for x in range(18)]
polydegree=int(sys.argv[1])
for d in datasets:
  X,y = getpolyxandy(trainfolder+d+'/train.csv',polydegree)
  clf = GridSearchCV(DecisionTreeRegressor(), {'max_depth':[x for x in range(10,50)]},n_jobs=40)
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=1)
  clf.fit(X_train, y_train)
  y_pred = clf.best_estimator_.predict(X_test)
  loss = scoreperdataset(y_pred,y_test)
  final_acc.append(loss)
  X_real,y_real = getpolyxandy(trainfolder+d+'/test.csv',polydegree,withy=False)
  y_real_pred = clf.best_estimator_.predict(X_real)
  pred_array[d+'.z'] = y_real_pred
  print('loss for {} is {}, maxdepth is {}'.format(d,loss, clf.best_estimator_.max_depth))
print('final avg loss is {:.5f}'.format( sum(final_acc)/len(final_acc)))
outdf = pd.DataFrame(pred_array)
outdf.to_csv('submission_{}_pd{}.csv'.format(trainfolder[:-1],polydegree),index_label='id')


# In[22]:

print("start to work on large datasets")
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
final_acc = []
pred_array = {}
trainfolder = 'large_datasets/'
datasets = [chr(ord('A') + x) for x in range(18)]
polydegree=int(sys.argv[1])
for d in datasets:
  X,y = getpolyxandy(trainfolder+d+'/train.csv',polydegree)
  clf = GridSearchCV(DecisionTreeRegressor(), {'max_depth':[x for x in range(10,50)]},n_jobs=20)
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=1)
  clf.fit(X_train, y_train)
  y_pred = clf.best_estimator_.predict(X_test)
  loss = scoreperdataset(y_pred,y_test)
  final_acc.append(loss)
  X_real,y_real = getpolyxandy(trainfolder+d+'/test.csv',polydegree,withy=False)
  y_real_pred = clf.best_estimator_.predict(X_real)
  pred_array[d+'.z'] = y_real_pred
  print('loss for {} is {}, maxdepth is {}'.format(d,loss, clf.best_estimator_.max_depth))
print('final avg loss is {:.5f}'.format( sum(final_acc)/len(final_acc)))
outdf = pd.DataFrame(pred_array)
outdf.to_csv('submission_{}_pd{}.csv'.format(trainfolder[:-1],polydegree),index_label='id')




