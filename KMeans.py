import pandas as pd
import numpy as np
import copy

from ClusterEntity import ClusterEntity
from Datapoint import Datapoint

class KMeans:
    def __init__(self, df: pd.DataFrame, k: int = None):
        self.__validateInputs(k,df)

        self.centroids = []
        self.datapoints = []
        self.k = k
        self.__df = df
        self.inertia = 0
        self.iterations = 0

        # When K hasn't been manually set, automatically assign K
        if self.k is None:
            self.k = self.findOptimalK()

        self.__setupAlgorithm()

    '''
    Public Methods for user interaction
    '''

    def findOptimalK(self):
        """
        runs the kmeans algorithm with differing values of k
        finds the average silhouette coefficient of each datapoint
        and produces the value of K with the highest coefficient.

        :returns Optimal value of K
        """
        silhouetteCoefficients = []
        for x in range(2, 11):
            kmeans = KMeans(self.__df, x)
            silhouetteCoefficients.append(self.__silhouette_coefficient(kmeans.run()))
        optimalCoefficient = max(silhouetteCoefficients)
        return silhouetteCoefficients.index(optimalCoefficient)+2


    def run(self):
        """
        This is the main method of the K-Means algorithm where the bulk of the computation occurs.
        it is responsible for associating Data Points to Centroids and for placing Centroids based on the average
        location of its associated Data Points. This occurs until the Convergence of Centroids or until the maximum number
        of iterations occurs.
        """
        '''
        PreviousCentroids is used to keep track of the state of self.centroids in the previous iteration so that when 
        convergence has occurred, the algorithm does not continue iterating.
        At the start of each iteration, a deep copy of self.centroids is created so that the state of PreviousCentroids 
        is not altered when the objects contained within self.centroids are changed (objects are copied by reference by default)
        '''

        MAX_ITERATIONS = 100
        for x in range(MAX_ITERATIONS):
            previousCentroids = copy.deepcopy(self.centroids)
            '''
            This loop is responsible for calculating the squared Euclidean Distance between all data points and all centroids.
            For each data point, it calculates the distance to all centroids and links itself to the centroid closest to it.
            '''
            self.inertia = 0
            for datapoint in self.datapoints:
                currentCentroidDistances = {}
                for centroid in self.centroids:
                    #create a key value pair, linking the current centroid id to the distance from the current data point
                    squaredDistance = self.__squaredEuclideanDistance(datapoint, centroid)
                    currentCentroidDistances[centroid.centroid_id] = squaredDistance
                    self.inertia += squaredDistance


                #set the datapoint's centroid id to the id of the centroid with the shortest distance
                datapoint.centroid_id = min(currentCentroidDistances, key=lambda distance: currentCentroidDistances[distance])

            '''
            This loop is responsible for setting each centroid to the average location of all its associated datapoints.
            For each centroid, it determines whether each datapoint is associated to it, then finds the average of each dimension 
            of those datapoints and sets its respective dimensions to those values. 
            '''
            for centroid in self.centroids:
                associatedDataPointsToCurrentCentroid = []
                for datapoint in self.datapoints:
                    if datapoint.centroid_id == centroid.centroid_id:
                        associatedDataPointsToCurrentCentroid.append(datapoint)
                if len(associatedDataPointsToCurrentCentroid) != 0:
                    #sets each dimension of each centroid to the average of the respective associated datapoints' dimensions
                    for i in range(len(centroid.dimensions)):
                        centroid.dimensions[i] = sum(dp.dimensions[i] for dp in associatedDataPointsToCurrentCentroid) / len(associatedDataPointsToCurrentCentroid)

            '''
            Convergence Check
            uses numpy operations to count how many previousCentroids dimensions are the same as the current centroids dimensions
            '''
            counter = 0
            for i in range(len(previousCentroids)):
                isSame = previousCentroids[i].dimensions == self.centroids[i].dimensions
                if np.all(isSame):
                    counter += 1
            '''
            Successful Completion of Algorithm
            When convergence has occurred, mean intra/inter cluster distance is calculated for each datapoint and 
            the results are returned
            '''
            if counter == self.k:

                self.__calculate_mean_intra_cluster_distance()

                self.__calculate_mean_inter_cluster_distance()
                self.iterations = x

                results = [copy.deepcopy(self.k), copy.deepcopy(self.iterations), copy.deepcopy(self.inertia), copy.deepcopy(self.centroids), copy.deepcopy(self.datapoints)]
                return results
        raise ValueError("Centroids never converged, maximum iterations reached")

    def outputToConsole(self):
        """
        This method helps visualise the result to the user by printing the results to the console
        without them having to understand the output from run().
        """
        print("\nAll Data Points:")
        counter = 0
        for datapoint in self.datapoints:
            counter += 1
            print(f"No. {counter}- dimensions:{datapoint.dimensions}\n"
                  f"         associated centroid: {datapoint.centroid_id}\n")
        print("\nAll Centroids:")
        for centroid in self.centroids:
            print(f"No: {centroid.centroid_id}\n   dimensions:{centroid.dimensions}")
        print(f"\nNumber of Centroids (K): {self.k}")
        print(f"Number of Iterations: {self.iterations}")
        print(f"Total Inertia: {self.inertia}")

    def getDataframes(self):
        """
        This method returns 2 pandas dataframes containing the data about the current state of the datapoints
        and the centroids. This is to make the output more friendly to users so that the data can be used for
        further processing or visualisation without them having to understand the output from run()
        """
        dfCopy = copy.deepcopy(self.__df)

        dfCopy["Associated_Centroid"] = np.nan
        dfCopy["Silhouette_Coefficient"] = np.nan

        for i, dp in enumerate(self.datapoints):
            #code adapted from GeeksForGeeks (2021)
            dfCopy.at[i, "Associated_Centroid"] = dp.centroid_id
            dfCopy.at[i, "Silhouette_Coefficient"] = dp.calculateSilhouetteCoefficient()
            #end of adapted code

        rows = []
        for centroid in self.centroids:
            dims = list(centroid.dimensions) + [centroid.centroid_id]
            rows.append(dims)

        cols = list(self.__df.columns)
        cols.append("Centroid_No")

        centroidsDF = pd.DataFrame(rows, columns=cols)
        return dfCopy, centroidsDF


    '''
    Private Methods for internal functionality
    '''

    @staticmethod
    def __validateInputs(k, df):
        # ensures data has been inputted correctly into algorithm
        if (not isinstance(k, int)) and (k is not None):  # k type check
            raise TypeError(f"Expected k to be int, got {type(k).__name__}")
        if not isinstance(df, pd.DataFrame):  # df type check
            raise TypeError(f"Expected df to be pandas DataFrame, got {type(df).__name__}")
        if len(df.columns) < 2:  # df too small check
            raise ValueError("Input data frame has too little dimensions, ensure data frame has 2 columns minimum")

        # code adapted from Saturn Cloud (2023)
        # checks data frame is fully numeric (int, float, etc. )
        numeric_columns = df.select_dtypes(include=np.number).columns
        if not len(numeric_columns) == len(df.columns):
            # end of adapted code
            raise ValueError("Data Frame expected to be fully numeric, other data types found")

    @staticmethod
    def __squaredEuclideanDistance(a: ClusterEntity, b: ClusterEntity):
        """
            This method calculates the Squared Euclidean Distance (S/ED) between a data point and a centroid
             in both 2D and 3D.

            The distance is still squared to maximise performance of the algorithm (square root can be
            resource intensive when constantly being performed). When determining the closest centroid
            to a data point, comparing SED yields the same outcome as it would with true ED.
        """
        difference = a.dimensions - b.dimensions
        return (difference**2).sum()

    def __euclideanDistance(self, a: ClusterEntity, b: ClusterEntity):
        """
        This method calculates the true Euclidean Distance between two datapoints/centroids.
        This is only used when calculating Intra-Cluster Distance and Inter-Cluster distance so that
        the outcome of the total silhouette coefficient is consistent with the known typical values
        of it.
        """
        return np.sqrt(self.__squaredEuclideanDistance(a, b))

    def __createDataPoints(self, row):
        #create a Datapoint object and append it to datapoints
        self.datapoints.append(Datapoint(row))


    def __setupAlgorithm(self):
        #create k number of centroids, using random points in the existing data as starting points
        for i in range(self.k):
            randomDFRow = self.__df.sample(n =1, random_state=20)
            dimensions = []
            for j in range(randomDFRow.size):
                dimensions.append(randomDFRow.iloc[0,j])
            self.centroids.append(ClusterEntity(dimensions, i))
        # code adapted from GeeksForGeeks (2024)
        # apply a method to each row in the dataframe to populate datapoints
        self.__df.apply(self.__createDataPoints, axis=1)
        # end of adapted code

    @staticmethod
    def __silhouette_coefficient(results):
        """
        Explanation: the Silhouette Coefficient is a measure in the K-Means algorithm to determine the optimal value
        of K, deriving from the expression:

        s = (b - a) / max(a, b)

        Where:
        a = intra-cluster distance (or distance from a datapoint to all datapoints in its cluster)
        b = inter-cluster distance (or distance from a datapoint to all datapoints in the neighbouring cluster)
        s = Silhouette Coefficient

        The Silhouette Coefficient is calculated for each datapoint, this method finds the average Coefficient for
        all datapoints from a result of the K-Means algorithm.

        :param results - dictionary returned from the run method
        :return: - mean silhouette coefficient of all data points from a run of the algorithm
        """
        datapoints = results[4]
        return sum(dp.calculateSilhouetteCoefficient() for dp in datapoints) / len(datapoints)

    def __calculate_mean_intra_cluster_distance(self):
        """
        Yadav (2023) discusses intra cluster distance to be 'the average distance between data points within the same cluster'
        and essentially measures the 'compactness of datapoints within a cluster', the code in this method is original,
        but based on this definition.
        """
        # calculate and set the intra cluster distance for all datapoints
        for datapoint in self.datapoints:
            totalIntraClusterDistance = 0
            noIntraDatapoints = 0
            #find sum of Euclidean distances between datapoints in the same centroid
            for intraDatapoint in self.datapoints:
                if datapoint != intraDatapoint and datapoint.centroid_id == intraDatapoint.centroid_id:
                    noIntraDatapoints += 1
                    totalIntraClusterDistance += self.__euclideanDistance(datapoint, intraDatapoint)
            #set Mean Intra Cluster Distance for each datapoint to the average distance
            if noIntraDatapoints > 0:
                datapoint.meanIntraClusterDistance = totalIntraClusterDistance / noIntraDatapoints
            else:
                datapoint.meanIntraClusterDistance = 0

    def __calculate_mean_inter_cluster_distance(self):
        """
        Yadav (2023) discusses inter cluster distance to be 'the distance between the centroids (mean or center points)
        of the clusters or as the minimum distance between data points in different clusters.'. Therefore, this code has
        interpreted this by finding the closest centroid to each centroid then finding the average distance to all the
        datapoints contained within that centroid.
        """
        # calculate and set an approximate inter cluster distance for all datapoints (based on closest centroid)
        for centroid in self.centroids:
            # find closest centroid to current centroid
            localCentroids = {}
            for interCentroid in self.centroids:
                if centroid != interCentroid:
                    localCentroids[self.__euclideanDistance(centroid, interCentroid)] = interCentroid.centroid_id

            lowestDistance = min(localCentroids)
            closestCentroidID = localCentroids[lowestDistance]

            #seperate datapoints depending on their associated centroid
            datapointsInCentroid = []
            datapointsInInterCentroid = []
            for datapoint in self.datapoints:
                if datapoint.centroid_id == closestCentroidID:
                    datapointsInInterCentroid.append(datapoint)
                if datapoint.centroid_id == centroid.centroid_id:
                    datapointsInCentroid.append(datapoint)

            #for each datapoint in each group, find the average distance between them and set the Mean Inter Cluster Distance
            for datapointA in datapointsInCentroid:
                sumDistance = 0
                for datapointB in datapointsInInterCentroid:
                    sumDistance += self.__euclideanDistance(datapointA, datapointB)
                datapointA.meanInterClusterDistance = sumDistance / len(datapointsInInterCentroid)