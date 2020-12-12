FROM tfp/tensorflow-probability
MAINTAINER Datmo devs <dev@datmo.io>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install --upgrade ipython && \
    pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        tf_agents \
        && \
    python -m ipykernel.kernelspec

# Install OpenCV
RUN apt-get update && apt-get install -y libopencv-dev python-opencv && \
    echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc


WORKDIR /workspace
RUN chmod -R a+w /workspace