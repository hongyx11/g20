import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization,Dropout

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


def getxandy(filename,withy=True):
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
  return x,y

f = os.listdir('nnmodel')
for fname in f:
  abspath = 'nnmodel/'+fname
  model = keras.models.load_model(abspath)
  Xtest,_ = getpolyxandy('small_datasets/A/test.csv',4,withy=False)
  ypred = model.predict(Xtest)
  print(ypred)