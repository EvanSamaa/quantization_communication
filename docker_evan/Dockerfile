FROM rocker/cuda:3.6.0

ENV CUDA=11.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-11.0 \
        cuda-cublas-11.0 \
        cuda-cufft-11.0 \
        cuda-curand-11.0 \
        cuda-cusolver-11.0 \
        cuda-cusparse-11.0 \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

## install as user to avoid venv issues later
USER rstudio
RUN pip3 install \
    h5py \
    pyyaml \
    requests \
    Pillow \
    tensorflow-gpu \
    tensorflow-probability \
    tf_agents \
    scipy \
    numpy \
    pandas \
    matplotlib \
    keras \
    --no-cache-dir
USER root
RUN install2.r reticulate tensorflow keras