from keras.layers import core, wrappers
from keras.models import Sequential
import numpy as np

try:
  model = Sequential()
  model.add(wrappers.TimeDistributed(core.Dropout(.5), input_shape=(3, 2)))
  model.compile(optimizer='rmsprop', loss='mse')
  model.predict(np.random.random((10, 3, 2)))
except Exception as e:
  print(str(e))
