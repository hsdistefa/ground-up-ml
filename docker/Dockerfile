FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        python \
        python-dev \
        python-pip \
        python-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    matplotlib \
    numpy \
    scipy \
    sklearn

COPY supervised /ground-up-ml/supervised

COPY unsupervised /ground-up-ml/unsupervised

WORKDIR "/ground-up-ml"

# Use agg instead of default Xwindows for matplotlib
ENV MPLBACKEND="agg"
