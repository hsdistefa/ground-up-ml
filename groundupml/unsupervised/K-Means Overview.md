# K-Means Clustering
*Unsupervised Learning*

### Overview
**K-Means Clustering** is a form of unsupervised learning. The goal of unsupervised learning is to classify data into meaningful clusters without any labels--it is attempting to say "these should be in one group and those should be in another" without being given any clue as to what a "group" should be.

How is it decided how many of these groups the data should be placed? Some unsupervised machine learning algorithms, such as **DBSCAN**, decide themselves; however, in K-Means the number of groups needs to be specified beforehand. That's what **K** stands for--the number of groups into which to separate the data, where each group is centered around a **mean**, as in the average of the points in that group.

Here is an example of 3-Means Clustering in action:

![Imgur](https://i.imgur.com/pdQht8u.gif)

You can see that each of the points in the dataset is placed into one of the three groups red, yellow, or blue and that each color has a larger point towards its center that moves with each iteration. These larger points are the mean of all the other points of the same color, called **centroids** because they are at the weighted center of their respective group. The algorithm works by first deciding where each centroid should start (see [Initialization](#initialization)) and assigning each point in the dataset to the group of the closest centroid. Then, each centroid is recalculated by taking the mean of all the points assigned to it. This moves the centroids, which means that the groups can again be reassigned to the nearest centroid. This continues until there is no change from the previous iteration or until a specified maximum number of iterations is reached.

At the end you have K clusters separated linearly that tend to span a roughly similar amount of space. You may have noticed that the moving lines represent the boundaries between the clusters--interestingly, these separating lines form a [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram).

### Initialization

The result of K-Means is very dependant on how the initial centroids are chosen, so a good initialization method can significantly improve the outcome of the algorithm. 

Here is an example of what can happen with poorly chosen initial centroids:

![Imgur](https://i.imgur.com/zDLdruh.gif)

As you can see, although there are four clearly defined clusters that are about the same size, because of how the centroids (here shown as stars) were initially placed the algorithm decides to split the top-left cluster into two smaller ones while combining the lower two clusters into one big group. 

So how do we avoid this? Let's go over a few possible initialization methods.

**Forgy Method**
The Forgy Method is the simplest way to initialize, where the starting centroids are chosen completely at random from the points in the dataset. This is generally not ideal unless K is very very large, making other initialization methods impractical.

**Random Partition**
Random partition assigns each point to a cluster at random, then computes the initial centroids from those clusters. This tends to initialize centroids that are near the center of all the points, which makes it more likely for them to end up nearer to the optimal solution and would in fact be almost guaranteed to work well on a dataset like the one above.

**K-Means++**
In K-Means++ first centroid is chosen at random from the points in the dataset, then each successive centroid is chosen in a way such that points farther from the other centroids are more likely to be chosen. This makes the algorithm perform provably competitive to the optimal solution in both speed and performance!

### Strengths and Weaknesses of K-Means Clustering

**Strengths**:
- Relatively simple and easy to implement
- Fast and efficient on larger datasets

**Weaknesses**:
- K value needs to be provided
- Final result dependant on the initialization
- Assumes that clusters are roughly spherical, which is often not the case
- Does not work well with clusters of varying size or density

K-Means Clustering is best used on datasets when you know how many groups you want and each of those groups has similar variability, such as reducing the color palette of an image to a smaller fixed number of colors. If you have a dataset with characteristically different variability, and isn't too large, it may be better to look at a heirarchical unsupervised learning algorithm.

