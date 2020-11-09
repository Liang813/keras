import numpy as np
from keras.layers import Input, Conv1D
from keras.models import Model

# Get model:
inp = Input(shape=(None, 16))
out = Conv1D(2, 3, padding='same')(inp)

model = Model(inputs=inp, outputs=out)
model.compile('adam', 'mse')

for len_ in (100, 200):
    x = np.random.random((4, len_, 16))
    model.predict(x)
