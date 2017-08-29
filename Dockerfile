FROM nlubock/cuda-theano

LABEL authors="Dirk von Gruenigen"

# Clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libtcmalloc-minimal4

WORKDIR /PA-ML-2016/
ADD /requirements.txt /PA-ML-2016/requirements.txt
RUN pip install cython
RUN pip install -r /PA-ML-2016/requirements.txt
RUN python -m nltk.downloader punkt

RUN ["/bin/bash"]
