FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Jan Deriu <deri@zahw.ch>

ARG THEANO_VERSION=0.9.0
ARG TENSORFLOW_VERSION=1.1.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.0.4
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest

# Install some dependencies
RUN apt-get update && apt-get install -y \
        bc \
        build-essential \
        cmake \
        curl \
        g++ \
        gfortran \
        git \
        libffi-dev \
        libfreetype6-dev \
        libhdf5-dev \
        libjpeg-dev \
        liblcms2-dev \
        libopenblas-dev \
        liblapack-dev \
        libpng12-dev \
        libssl-dev \
        libtiff5-dev \
        libwebp-dev \
        libzmq3-dev \
        nano \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget \
        zlib1g-dev \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
    update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

RUN apt-get upgrade

RUN wget http://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh

RUN ls

RUN sh -c '/bin/echo -e "\nyes\n" |bash Anaconda3-4.4.0-Linux-x86_64.sh'

ENV PATH "$PATH:/root/anaconda3/bin"

RUN conda install numpy scipy mkl nose sphinx

RUN conda install theano pygpu

#Install Theano
RUN echo "[global]\ndevice=cuda0\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
        \n[lib]\ncnmem=0.95 \
        \n[nvcc]\nfastmath=True" \
    > /root/.theanorc

# Install TensorFlow
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl

# Install Keras
RUN pip install --no-cache-dir --upgrade keras

# Fix problems with the scipy.signal package beeing missing..
RUN echo 'y' | pip uninstall scipy
RUN pip install scipy --no-cache-dir

RUN mkdir -p /root/.keras
RUN echo "{\"floatx\": \"float32\",\n\"epsilon\": 1e-07,\n\"backend\": \"theano\",\n\"image_data_format\": \"channels_last\"}" \
    > /root/.keras/keras.json

RUN pip install --no-cache-dir --upgrade nltk
RUN pip install --no-cache-dir --upgrade tqdm
RUN pip install --no-cache-dir --upgrade gensim

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

CMD python3 main.py -c config_vae.json

WORKDIR /PA-ML-2016/

ADD /requirements.txt /PA-ML-2016/requirements.txt
RUN pip install cython
RUN pip install -r /PA-ML-2016/requirements.txt
RUN pip uninstall numpy && pip install -U numpy
RUN python -m nltk.downloader punkt

RUN ["/bin/bash"]
