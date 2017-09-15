# Ground Up Machine Learning

A set of bare bones implementations of machine learning algorithms intended to be explanatory and used as a learning reference.

## Installation
    $ git clone https://github.com/hsdistefa/ground-up-ml
    $ cd ground-up-ml
    $ pip install -r requirements.txt

### Using Docker
[Install Docker](https://docs.docker.com/engine/installation/)

Get the docker image and run, replacing /your/folder with desired folder to save jupyter notebooks in:
``` sh
    $ docker pull hsdistefa/ground-up-ml
    $ docker run -it -p 8888:8888 -v /your/folder/notebooks:/notebooks ground-up-ml
```

Copy and paste the URL for the jupyter server into your browser, it should be of the form:
``` sh
    http://localhost:8888/?token=...
```

