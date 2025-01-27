import numpy as np
''' 
This class is used to represent Centroids in the K-Means algorithm but is also the parent class to 
Datapoints - hence its more general name
In any case:
- dimensions represents a single Data Point/Centroid's values (a list of a single row of the input data frame to
the K-Means algorithm)

In the case that ClusterEntity is being used to represent a Centroid:
- centroid_id refers to a unique identifier for that centroid.

In the case that ClusterEntity is being used to represent a Data Point:
- centroid_id refers to the unique identifier of the centroid closest to it.
'''
class ClusterEntity:
    def __init__(self, dimensions, centroid_id = None):
        self.dimensions = np.array(dimensions)
        self.centroid_id = centroid_id