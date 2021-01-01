
from keras import Input
from keras.backend import int_shape
from keras.layers import Conv2D, Conv2DTranspose
try:
  conv = Conv2D(16, 3, strides=2, padding='same')
  transpose_conv = Conv2DTranspose(1, 3, strides=2, padding='same')

  input_a = Input(shape=(23, 23, 1))
  x_conv_a = conv(input_a)
  x_transpose_a = transpose_conv(x_conv_a)

  print("(a) Input shape: {}".format(int_shape(input_a)))
  print("(a) Shape after convolution: {}".format(int_shape(x_conv_a)))
  print("(a) Shape after transposed convolution: {}".format(int_shape(x_transpose_a)))
  print()

  input_b = Input(shape=(24, 24, 1))
  x_conv_b = conv(input_b)
  x_transpose_b = transpose_conv(x_conv_b)

  print("(b) Input shape: {}".format(int_shape(input_b)))
  print("(b) Shape after convolution: {}".format(int_shape(x_conv_b)))
  print("(b) Shape after transposed convolution: {}".format(int_shape(x_transpose_b)))
except Exception as e:
  print(str(e))
