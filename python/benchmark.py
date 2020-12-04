from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def benchmark_tftrt(saved_model_loaded,batched_input):

    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []

    for i in range(N_warmup_run):
      labeling = infer(batched_input)

    for i in range(N_run):
      start_time = time.time()
      labeling = infer(batched_input)
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))

def predict_tftrt(saved_model_loaded):

    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    
    for i in range(4):
        img_path = './data/img%d.JPG'%i
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = tf.constant(x)
        labeling = infer(x)
        preds = labeling['predictions'].numpy()
        print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

modelToUse = sys.argv[1]
print("Using model: ")
print(modelToUse)


print('Loading model')
saved_model_loaded = tf.saved_model.load(modelToUse, tags=[tag_constants.SERVING])
print("Calling predict_tftrt")
predict_tftrt(saved_model_loaded)

batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

for i in range(batch_size):
  img_path = './data/img%d.JPG' % (i % 4)
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  batched_input[i, :] = x
batched_input = tf.constant(batched_input)
print('batched_input shape: ', batched_input.shape)
print("benchmarking")
benchmark_tftrt(saved_model_loaded,batched_input)
