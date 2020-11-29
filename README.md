# tf-trt


This example demontrates how to use the TensorRT integration with TensorFlow 2.x (tf-trt) and the generation of tf-trt fp32, fp16, and int8 models on the Jetson Xavier NX.

## Setup 
You'll need to be using Jetpack 4.4.x

### Protobuf
There appears to be an issue with the python implementation protobuf (see https://jkjung-avt.github.io/tf-trt-revisited).  The current workaround is to build and install a C++ based implemenation. The script install_protobuf-3.13.0.sh will download, build, and install protobuf 3.13.0.  If a later version is needed, the script can be easily updated.  Run the script with the command:
```
sh install_protobuf-3.13.0.sh
```
The script will take some time to complete.  

Now you'll need to run the following commands
```
cd /usr/local/lib/python3.6/dist-packages/protobuf-3.13.0-py3.6-linux-aarch64.egg/google
sudo cp -R /usr/local/lib/python3.6/dist-packages/protobuf-3.13.0-py3.6-linux-aarch64.egg/google/protobuf /usr/local/lib/python3.6/dist-packages/google/
```

Once done, reboot your NX.

### TensorFlow 2.x
You'll want to make sure TensorFlow 2.x is installed; see https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html for details.

### Jetson Stats

Jetson-stats is a package to monitoring and control your NVIDIA Jetson [Xavier NX, Nano, AGX Xavier, TX1, TX2] Works with all NVIDIA Jetson ecosystem.  
Jtop can be used to monitor your NX's memory, cpu, gpu, etc.,  along with being able to clear the buffers as needed.

Follow the instructions at https://github.com/rbonghi/jetson_stats#jetson_release to install.  

## Overview
This is a very simple image classification example based on https://github.com/tensorflow/tensorrt/tree/master/tftrt/examples/image-classification. You'll learn how to convert a Keras model to three tf-trt models, a fp32, fp16, and int8.  A simple set of test images will be used to both validate and benchmark both the native model and the three tf-trt ones.

### Scripts
tbd

Download the test images with the following commands

```
mkdir ./data
wget  -O ./data/img0.JPG "https://d17fnq9dkz9hgj.cloudfront.net/breed-uploads/2018/08/siberian-husky-detail.jpg?bust=1535566590&width=630"
wget  -O ./data/img1.JPG "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
wget  -O ./data/img2.JPG "https://www.artis.nl/media/filer_public_thumbnails/filer_public/00/f1/00f1b6db-fbed-4fef-9ab0-84e944ff11f8/chimpansee_amber_r_1920x1080.jpg__1920x1080_q85_subject_location-923%2C365_subsampling-2.jpg"
wget  -O ./data/img3.JPG "https://www.familyhandyman.com/wp-content/uploads/2018/09/How-to-Avoid-Snakes-Slithering-Up-Your-Toilet-shutterstock_780480850.jpg"
```


### Notebook




https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples/image-classification/NGC-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb

https://jkjung-avt.github.io/tf-trt-revisited/ -> update for latest protobuff

Split up into multiple apps and or a new ipynb file
