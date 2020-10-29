from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization
K.set_image_data_format('channels_first')
input = Input(shape=(3, 1001, 1001), dtype='float32')
x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=1)(x)
