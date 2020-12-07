

# download the dataset and save it in the same dir with this file.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn import neighbors
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
kf = KFold(n_splits=10)
# import scikit learn
from sklearn import linear_model,svm
from sklearn.preprocessing import PolynomialFeatures

def getxandy(filename):
  with open(filename,'r') as f:
    df = pd.read_csv(f,index_col=0)
    x = np.zeros((len(df.x),2))
    y = np.zeros((len(df.x)))
    x[:,0] = df.x.to_numpy()
    x[:,1] = df.y.to_numpy()
    y = df.z.to_numpy()
  return x,y

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

def scoreperdataset(y_pred,y_test):
  return np.sqrt(np.sum(np.square(y_pred - y_test))/len(y_pred))

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




trainfolder = 'small_datasets/'
polydegree=4
X,y = getpolyxandy(trainfolder+'L'+'/train.csv',polydegree)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=1)

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 20} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)
# create model
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=X.shape[1], kernel_initializer="uniform"))
model.add(Dense(128, activation="relu", input_dim=256, kernel_initializer="uniform"))
model.add(Dense(64, activation="relu", input_dim=128, kernel_initializer="uniform"))
model.add(Dense(32, activation="relu", input_dim=64, kernel_initializer="uniform"))
model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
model.summary()
# Compile model
model.compile(loss='mse', optimizer='adam')

# Fit the model
model.fit(X_train, y_train, epochs=500, batch_size=50,validation_split=0.2,verbose=2)
#y_pred = model.predict(X_test)
