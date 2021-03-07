from keras.layers import Add, Input
import tsensor

a = Input(shape=(10, 3))
b = Input(shape=(10, 3))
add_layer = Add()
c = add_layer([a, b])

with tsensor.clarify():
  assert add_layer.output_shape == (None, 10, 3)
