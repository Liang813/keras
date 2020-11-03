from keras.layers.wrappers import Bidirectional
from keras.layers import LSTM, Dense, Input
from keras.models import Model
import numpy as np
data=np.random.rand(1,5,10)
label=np.asarray([[1,0,0]])

inp = Input(shape=data.shape[1:])
x = Bidirectional(LSTM(units=32,recurrent_dropout=0.5))(inp)
x = Dense(3,activation='softmax')(x)
model = Model(input=inp, output=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(data, label, epochs=1)
