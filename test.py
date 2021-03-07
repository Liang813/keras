from keras.layers import Add, Input
import tsensor
try:
  a = Input(shape=(10, 3))
  b = Input(shape=(10, 3))
  add_layer = Add()
  with tsensor.clarify():
    c = add_layer([a, b])
  assert add_layer.output_shape == (None, 10, 3)
except AssertionError as e:
  print(e)
