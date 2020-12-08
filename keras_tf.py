import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization,Dropout



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


def writeresults2(X_test,y_pred,f):
    data = {}
    data['x'] = X_test[:,0]
    data['y'] = X_test[:,1]
    data['z_pred'] = y_pred
    df = pd.DataFrame(data)
    if not os.path.isdir('results/'):
      os.mkdir('results')
    df.to_csv('results/'+f+'.csv',float_format="%.6f")


trainfolder = 'small_datasets/'
polydegree=4

datasets = [chr(ord('A') + x) for x in range(18)]

for d in datasets:
  X,y = getpolyxandy(trainfolder+d+'/train.csv',polydegree)
  model = Sequential()
  model.add(Dense(256, activation="relu", input_dim=X.shape[1], kernel_initializer="uniform"))
  model.add(Dense(128, activation="relu", input_dim=256, kernel_initializer="uniform"))
  model.add(Dense(64, activation="relu", input_dim=128, kernel_initializer="uniform"))
  model.add(Dense(32, activation="relu", input_dim=64, kernel_initializer="uniform"))
  model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
  model.summary()
  # Compile model
  opt = tf.keras.optimizers.Adam(learning_rate=0.005)
  model.compile(loss='mse', optimizer=opt)
  # Fit the model
  output = model.fit(X, y, epochs=2000, batch_size=1024,validation_split=0.2,verbose=2)
  val_loss = output.history['val_loss']
  X_test,_ = getpolyxandy(trainfolder + d + '/test.csv', polydegree, withy=False)
  X_testorig,_ = getxandy(trainfolder+d+'/test.csv', withy=False)
  y_pred = model.predict(X_test)
  print('last 10 val loss is ', val_loss[-10:])
  print(y_pred.shape)
  writeresults2(X_testorig, y_pred.flatten(), "keras_{}_{:.3f}_".format(d,val_loss[-1]))
  if not os.path.isdir('nnmodel'):
    os.mkdir('nnmodel')
  model.save("nnmodel/keras_{}_{:.3f}".format(d,val_loss[-1]))
