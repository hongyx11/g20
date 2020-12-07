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
from sklearn.metrics import mean_squared_error
import struct
import argparse
from scipy.stats import pearsonr,spearmanr
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sys
K = KFold(n_splits=10)

if len(sys.argv) != 2:
  print("you need specify poly degree")

def check_mse_corr(array_1, array_2):
        try:
                m = np.sqrt(mse(array_1, array_2))
        except Exception as e:
                m = None

        r, _ = pearsonr(array_1, array_2)
        spr, _ = spearmanr(array_1, array_2)

        return (m, r, spr)

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

param_grid = {
    'max_depth':[x for x in range(10,50)],
    #'verbose': 1,
    #'n_estimators': [100, 200, 300, 1000]
    'n_estimators': [500]
}
accuracy=[]
ypred_list = {}
datapath='data/'
dataset = [chr(ord('A') + x) for x in range(18)]
polydegree=int(sys.argv[1])
for data in dataset:
  print('start new data')
  X,y = getpolyxandy(datapath+data+'/train.csv',polydegree)
  clf = GridSearchCV(GradientBoostingRegressor(), param_grid = param_grid,n_jobs=64, verbose=1)
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=1) 
  clf.fit(X_train, y_train)
  y_pred = clf.best_estimator_.predict(X_test)
  mse, r, spr = check_mse_corr(y_pred,y_test)
  accuracy.append(mse)
  X_real,y_real = getpolyxandy(datapath+data+'/test.csv',polydegree,withy=False)
  y_real_pred = clf.best_estimator_.predict(X_real)
  ypred_list[data+'.z'] = y_real_pred
  print('mse for {} is {}, maxdepth is {}'.format(data, mse, clf.best_estimator_.max_depth))
  print('pearsonr for {} is {}, maxdepth is {}'.format(data, r, clf.best_estimator_.max_depth))
  print('spearmanr for {} is {}, maxdepth is {}'.format(data, spr, clf.best_estimator_.max_depth))

print('final avg mse is {:.5f}'.format( sum(accuracy)/len(accuracy)))

outdf = pd.DataFrame(ypred_list)
outdf.to_csv('submission_{}_pd{}.csv'.format(datapath[:-1],polydegree),index_label='id')
