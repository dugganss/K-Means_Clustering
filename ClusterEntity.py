import numpy as np
''' TODO: edit this class to support n dimensions, edit comment
This class is used generically to represent all points in the K-Means Algorithm.
In any case:
- x refers to the x coordinate of the item.
- y refers to the y coordinate of the item.
- z refers to the z coordinate of the item if the data set is 3-Dimensional.

In the case that ClusterEntity is being used to represent a Centroid:
- centroid_id refers to a unique identifier for that centroid.

In the case that ClusterEntity is being used to represent a Data Point:
- centroid_id refers to the unique identifier of the centroid closest to it.

Justification:
Both modules would be identical if a class was made for each case.
'''
#class ClusterEntity:
#    def __init__(self, x , y, z = None, centroid_id = None):
#        self.x = x
#        self.y = y
#        self.z = z
#        self.centroid_id = centroid_id

class ClusterEntity:
    def __init__(self, dimensions, centroid_id = None):
        self.dimensions = np.array(dimensions)
        self.centroid_id = centroid_id