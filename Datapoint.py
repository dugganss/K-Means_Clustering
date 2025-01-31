from ClusterEntity import ClusterEntity

'''
This class represents a Datapoint in the K-Means algorithm

it inherits from ClusterEntity which provides the dimensions and centroid_id attributes

In this context: 
- dimensions refers to the single row of the DataFrame that a Datapoint can represent.
- centroid_id refers to the centroid_id of the ClusterEntity that this datapoint is associated with.

New attributes and methods:
- meanIntraClusterDistance- refers to the average euclidean distance between datapoints with the same centroid_id
- meanInterClusterDistance- refers to the average euclidean distance between datapoints with the centroid_id 
closest (by euclidean distance) to the current datapoint's centroid_id.
- calculateSilhouetteCoefficient() calculates the silhouette coefficient for the current datapoint using the inter/intra 
cluster distances, using the formula:

    s = (b - a) / max(a, b)
    
    where: 
    - a = meanIntraClusterDistance
    - b = meanInterClusterDistance
    - s = Silhouette Coefficient
    
(Silhouette Coefficient is used to represent the strength of grouping in the clusters which helps inform the best 
value of K)
'''

class Datapoint(ClusterEntity):
    def __init__(self, dimensions, centroid_id = None):
        super().__init__(dimensions, centroid_id )
        self.meanIntraClusterDistance = 0
        self.meanInterClusterDistance = 0


    def calculateSilhouetteCoefficient(self):
        #formula adapted from GeeksForGeeks (2019)
        return (self.meanInterClusterDistance - self.meanIntraClusterDistance)/ max(self.meanIntraClusterDistance,self.meanInterClusterDistance)
        #end of adapted formula
