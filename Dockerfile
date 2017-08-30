FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

LABEL authors="Dirk von Gruenigen"

WORKDIR /PA-ML-2016/

RUN mkdir -p /home/keras/.keras
ADD keras.json /home/keras/.keras/keras.json
ADD .theanorc /home/keras/.theanorc

ADD /requirements.txt /PA-ML-2016/requirements.txt
RUN pip install cython
RUN pip install -r /PA-ML-2016/requirements.txt
RUN python -m nltk.downloader punkt

RUN ["/bin/bash"]
