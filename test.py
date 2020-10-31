from keras.models import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(4, input_shape=(None, 7)))
