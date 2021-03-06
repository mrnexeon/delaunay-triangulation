## Delaunay triangulation Watson-Bowyer

This script is implementation of the [Bowyer–Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm) on Python for computing the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) from finite array of 2D points. New point can be added dynamically in runtime with mesh recalculation.  

<img src="result.png" alt="delaunay triangulation with circles" width="500"/>

It support either user input and random point generation. You can see or save GIF animation of step by step triangulation:

<img src="triangulation.gif" alt="delaunay triangulation animation" width="500"/>

Written and tested on Python 3 with `numpy` and `matplotlib`. 
