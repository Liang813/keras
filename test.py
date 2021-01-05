
from keras.applications import NASNetLarge
try:
  NASNetLarge(include_top=False, input_shape=(512,512,3))
except Exception as e:
  print(str(e))
