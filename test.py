
from keras.layers import Input, Convolution3D

try:
  a = Input(shape=(32, 32, 32, 3))
  b = Input(shape=(64, 64, 64, 3))

  conv = Convolution3D(16, 3, 3, 3, border_mode='same')
  conved_a = conv(a)

  # only one input so far, the following will work:
  assert conv.input_shape == (None, 32, 32, 32, 3)

  conved_b = conv(b)
  
except Exception as e:
  print(str(e))
