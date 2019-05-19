# Ground Up Machine Learning

A set of bare bones implementations of machine learning algorithms intended to be explanatory and used as a learning reference.

## Installation
    $ git clone https://github.com/hsdistefa/ground-up-ml
    $ cd ground-up-ml
    $ python setup.py install

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

## Example Usage

### PCA
PCA done on a dataset consisting of handwritten digit images. Each color corresponds to one of the digits 0 through 9.

The transformation does a good job of clustering each digit using only the first two principal components as features. Adding a third principal components would do even better and is a huge improvement on the original feature-space of 728 dimensions!

![PCA on Digits Dataset](https://user-images.githubusercontent.com/5913237/31640503-c3f59810-b293-11e7-8d9e-06e7d0e54c61.png)


### DBSCAN
DBSCAN on a moons dataset. The right figure shows how DBSCAN clusters the original dataset (left figure), with green and yellow corresponding to its guesses for the clustering and the purple being labeled as noise.

![DBSCAN on Moon-Shaped Data](https://user-images.githubusercontent.com/5913237/31640535-f8c2dc74-b293-11e7-8803-018a124c2812.png)
