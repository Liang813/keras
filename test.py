from keras.layers import Add, Input

a = Input(shape=(10, 3))
b = Input(shape=(10, 3))
add_layer = Add()
c = add_layer([a, b])
assert add_layer.output_shape == (None, 10, 3)

