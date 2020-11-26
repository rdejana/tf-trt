# tf-trt


This example demontrates how to use the TensorRT integration with TensorFlow 2.x (tf-trt) and the generation of tf-trt fp32, fp16, and int8 models on the Jetson Xavier NX.

## Setup 
You'll need to be using Jetpack 4.4.x

### Protobuf
There appears to be an issue with the python implementation protobuf (see https://jkjung-avt.github.io/tf-trt-revisited).  The current workaround is to build and install a C++ based implemenation. The script install_protobuf-3.13.0.sh will download, build, and install protobuf 3.13.0.  If a later version is needed, the script can be easily updated.  Run the script with the command:
```
sh install_protobuf-3.13.0.sh
```
The script will take some time to complete.  Once done, reboot your NX.

### TensorFlow 2.x
You'll want to make sure TensorFlow 2.x is installed; see https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html for details.


## Overview
This is a very simple image classification example based on https://github.com/tensorflow/tensorrt/tree/master/tftrt/examples/image-classification. You'll learn how to convert a Keras model to three tf-trt models, a fp32, fp16, and int8.  A simple set of test images will be used to both validate and benchmark both the native model and the three tf-trt ones.



https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples/image-classification/NGC-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb

https://jkjung-avt.github.io/tf-trt-revisited/ -> update for latest protobuff

Split up into multiple apps and or a new ipynb file
