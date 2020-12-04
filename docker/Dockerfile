FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3
RUN apt-get update && apt-get install python3-dev  -y 
RUN pip3 install -U pip
RUN pip3 install cffi setuptools pillow matplotlib notebook
RUN apt-get install git -y
RUN apt-get install sudo unzip -y 
WORKDIR /app
RUN git clone https://github.com/rdejana/tf-trt
WORKDIR /app/tf-trt
RUN sh install_protobuf-3.13.0.sh
RUN pip3 install -U jetson-stats
RUN  cp -R /usr/local/lib/python3.6/dist-packages/protobuf-3.13.0-py3.6-linux-aarch64.egg/google/protobuf /usr/local/lib/python3.6/dist-packages/google/
