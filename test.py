from keras.layers import Input, Conv2D
from keras.models import Model
import numpy as np

inputs = Input(shape=(10, 10, 1))
m = Conv2D(1, kernel_size=(3, 3), padding='valid', dilation_rate=(2, 2))(inputs)
model = Model(inputs=inputs, outputs=m)
model.summary()

x = np.random.random((1, 10, 10, 1)).astype(np.float32)
y = model.predict(x)
print("y.shape = " + str(y.shape))
