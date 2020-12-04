from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


model = ResNet50(weights='imagenet')



# Save the entire model as a SavedModel.
model.save('resnet50_saved_model')




