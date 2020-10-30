import tempfile
import os

from keras.layers import Input, GRU, CuDNNGRU, TimeDistributed
from keras.models import Model
import numpy as np

from tests.keras.layers.recurrent_test import units

input2 = Input((4, 6, 10))
output_cudnn = TimeDistributed(CuDNNGRU(units))(input2)
output = TimeDistributed(GRU(units, activation='hard_sigmoid', reset_after=True))(input2)
model_cudnn = Model(input2, output_cudnn)
model_plain = Model(input2, output)

_, fname = tempfile.mkstemp('.h5')
model_cudnn.save_weights(fname)
model_plain.load_weights(fname)
