from keras.layers import Input, Conv2D
from keras.models import Model
from keras.initializers import Constant
import numpy as np

kernel = np.arange(9).reshape((3, 3, 1, 1))
print(kernel.shape)

inputs = Input(shape=(10, 10, 1))
m = Conv2D(1, kernel_size=(3,3), padding='valid', dilation_rate=(2, 2), kernel_initializer=Constant(kernel))(inputs)
model = Model(inputs=inputs, outputs=m)
model.summary()


x = np.arange(100).reshape((1, 10, 10, 1)).astype(np.float32) 
y = model.predict(x)

print(np.squeeze(y))
print(y.shape)
