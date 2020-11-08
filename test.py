
import numpy as np

from keras.engine import Input, merge, Model

a = np.reshape(np.array([0, 0]), (1, 2, 1)) # zero vector has zero length
b = np.reshape(np.array([0.6, 0.8]), (1, 2, 1))

input_a = Input((2, 1))
input_b = Input((2, 1))

cos = merge([input_a, input_b], mode='cos', dot_axes=1)

model = Model(input=[input_a, input_b], output=[cos])
result = model.predict([a, b])
print(result)
# before fix: result  == [[[[ nan ]]]]
# after fix: result == [[[[ 0. ]]]]
