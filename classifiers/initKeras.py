import logging
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.models import load_model

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

tf.logging.set_verbosity(tf.logging.ERROR)
 
###################################
# TensorFlow wizardry
gpuLimit = 0.8
logging.warning(f"Limiting GPU to {gpuLimit}. Increase it in initKeras.py. Next is available GPUs")
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = gpuLimit
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
logging.warning(k.tensorflow_backend._get_available_gpus()) 
###################################