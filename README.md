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

### Random Forest
A Random Forest model trained on the Iris dataset. To do this 3 separate decision trees are constructed on a subset of the dataset containing randomly selected samples (with replacement, a.k.a ![bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))) and features. These simplified training sets help to reduce over-fitting which is a common issue found with individual decision trees.

After the forest is constructed, predictions can then be made by aggregating the results from all of these simpler decision trees to produce a result that can be better at generalizing than using a single decision tree.

![random_forest](https://user-images.githubusercontent.com/5913237/229948117-dad5c190-5fc7-472b-96b3-32a9e617f34a.png)

All three trees in the forest are shown above with decision nodes shown in blue, true branches in green, false branches in red, and class predictions in ovals. A prediction is made by traversing each decision tree according to the feature values in the test sample and choosing the class with the most 'votes.' 

For example, the first sample flower in our test set has the following features:

 - Sepal length of 5.6cm 
 - Sepal width of 2.9cm 
 - Petal length of 1.5cm
 - Petal width of 0.2cm 

Starting at the top of the first tree on the left, traverse the tree as follows:

1) Follow the red arrow (false) since sepal width is > 0.4
2) Follow the red arrow (false) again since sepal width is also > 1.7 
3) Finally because petal length is <= 4.8 we follow the green arrow (true) 
4) So this first decision tree predicts the flower is of the Versicolor variety

These steps are then repeated for the remaining two trees, yielding predictions of Versicolor and Virginica for the second and third trees respectively. Because 2/3 of the trees predicted Versicolor, that is the class that this random forest model predicts--which happens to be correct!

<!-- # TODO: Add Confusion Matrix of Results-->


### PCA
PCA done on a dataset consisting of handwritten digit images. Each color corresponds to one of the digits 0 through 9.

The transformation does a good job of clustering each digit using only the first two principal components as features. Adding a third principal components would do even better and is a huge improvement on the original feature-space of 728 dimensions!

![PCA on Digits Dataset](https://user-images.githubusercontent.com/5913237/31640503-c3f59810-b293-11e7-8d9e-06e7d0e54c61.png)


### DBSCAN
DBSCAN on a moons dataset. The right figure shows how DBSCAN clusters the original dataset (left figure), with green and yellow corresponding to its guesses for the clustering and the purple being labeled as noise.

![DBSCAN on Moon-Shaped Data](https://user-images.githubusercontent.com/5913237/31640535-f8c2dc74-b293-11e7-8803-018a124c2812.png)
