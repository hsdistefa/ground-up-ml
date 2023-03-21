FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY groundupml/ /groundupml/
COPY requirements.txt /groundupml/requirements.txt
COPY setup.py /groundupml/setup.py

RUN pip --no-cache-dir install \
    ipykernel \
    jupyter \
        && \
    python3 -m ipykernel.kernelspec

RUN pip --no-cache-dir install -r /groundupml/requirements.txt

# Set up jupyter notebook config
COPY docker/jupyter_config.py /root/.jupyter/

WORKDIR /groundupml

# Install groundupml modules
RUN python3 -m pip install -e \
    git+https://github.com/hsdistefa/ground-up-ml/#egg=groundupml

# Open a port for Jupyter notebook
EXPOSE 8888

# Use agg instead of default Xwindows for matplotlib
ENV MPLBACKEND="agg"

CMD jupyter notebook
