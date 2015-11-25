# TensorFlow and nodejs development container.
#

FROM debian:jessie
MAINTAINER Nikhil Kothari

# Setup OS and core packages
RUN echo "deb-src http://ftp.us.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y -q \
        curl wget unzip git vim build-essential ca-certificates pkg-config \
        python2.7 python-dev python-pip python-setuptools

# Setup Node.js
RUN mkdir -p /tools/node && \
    wget -nv https://nodejs.org/dist/v4.2.3/node-v4.2.3-linux-x64.tar.gz -O node.tar.gz && \
    tar xzf node.tar.gz -C /tools/node --strip-components=1 && \
    rm node.tar.gz

# Setup TensorFlow
RUN pip install --upgrade six
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Configuration
ENV PATH $PATH:/tools/node/bin

# Startup
ENTRYPOINT [ "/bin/bash" ]

