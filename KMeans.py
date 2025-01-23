import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt

from ClusterEntity import ClusterEntity

#TODO: Refactor code base to support N dimensions instead of hard-coding only 2 and 3 dimensions, will also have to change comments so that they explain the new code
'''
How you might be able to achieve above:
- change cluster entity to use a numpy array to store the coordinates, this will allow for any number of dimensions to be used
- within this class, wherever there are hardcoded methods to get or set coordinates, make it compatible with a numpy array instead
- the numpy library will be very helpful with the above, research how you can use it to calculate the ED 
- when recalculating centroids, because you have to find the average of all datapoints coordinates, place it into a 2d array and use numpy to find the average between teh dimensions
- remove visualisation in the output and move that somewhere else, that is not part of the algorithm. (it wont work with more than 3d anyway)
'''
class KMeans:
    centroids = []
    datapoints = []
    def __init__(self, k: int , df: pd.DataFrame):
        self.__validateInputs(k,df)

        if(len(df.columns) == 3):
            self.is3D = True
        else:
            self.is3D = False

        self.k = k
        self.df = df

        self.__setupAlgorithm()

    '''
    Public Methods for user interaction
    '''

    def run(self):
        '''
        This is the main method of the K-Means algorithm where the bulk of the computation occurs.
        it is responsible for associating Data Points to Centroids and for placing Centroids based on the average
        location of its associated Data Points. This occurs until the Convergence of Centroids or until the maximum number
        of iterations occurs.
        '''
        '''
        PreviousCentroids is used to keep track of the state of self.centroids in the previous iteration so that when 
        convergence has occured, the algorithm does not continue iterating.
        At the start of each iteration, a deep copy of self.centroids is created so that the state of PreviousCentroids 
        is not altered when the objects contained within self.centroids are changed (objects are copied by reference by default)
        '''
        previousCentroids = []
        MAX_ITERATIONS = 30
        for x in range(MAX_ITERATIONS):
            previousCentroids = copy.deepcopy(self.centroids)
            '''
            This loop is responsible for calculating the squared Euclidean Distance between all data points and all centroids.
            For each data point, it calculates the distance to all centroids and links itself to the centroid closest to it.
            '''
            for datapoint in self.datapoints:
                currentCentroidDistances = {}
                for centroid in self.centroids:
                    #create a key value pair, linking the current centroid id to the distance from the current data point
                    currentCentroidDistances[centroid.centroid_id] = self.__squaredEuclideanDistance(datapoint, centroid)
                    #print(f" data point: {i} to centroid: {self.centroids[j].centroid_id}: {self.__squaredEuclideanDistance(self.datapoints[i], self.centroids[j])}")

                #set the datapoint's centroid id to the id of the centroid with the shortest distance
                datapoint.centroid_id = min(currentCentroidDistances, key=lambda distance: currentCentroidDistances[distance])

            '''
            This loop is responsible for setting each centroid to the average location of all its associated datapoints.
            For each centroid, it determines whether each datapoint is associated to it, then finds the average x, y and z
            (if using 3 dimensional data) coordinates of those datapoints and sets its respective coordinates to those values.
            '''
            for centroid in self.centroids:
                associatedDataPointsToCurrentCentroid = []
                for datapoint in self.datapoints:
                    if(datapoint.centroid_id == centroid.centroid_id):
                        associatedDataPointsToCurrentCentroid.append(datapoint)
                if(len(associatedDataPointsToCurrentCentroid) != 0):
                    #TODO: no manual set of x y and z, use 2d array and numpy
                    centroid.x = sum(dp.x for dp in associatedDataPointsToCurrentCentroid) / len(associatedDataPointsToCurrentCentroid)
                    centroid.y = sum(dp.y for dp in associatedDataPointsToCurrentCentroid) / len(associatedDataPointsToCurrentCentroid)
                    if(self.is3D):
                        centroid.z = sum(dp.z for dp in associatedDataPointsToCurrentCentroid) / len(associatedDataPointsToCurrentCentroid)

            #print("data points with closest associated centroids:")
            #for i in range(len(self.datapoints)):
            #    print(f"x: {self.datapoints[i].x}, y: {self.datapoints[i].y}, z: {self.datapoints[i].z}, centroid id: {self.datapoints[i].centroid_id}")
            counter = 0
            for i in range(len(previousCentroids)):
                #TODO:need new method to check for convergence
                if(previousCentroids[i].x == self.centroids[i].x and previousCentroids[i].y == self.centroids[i].y and previousCentroids[i].z == self.centroids[i].z and x > 0):
                    counter +=1

            #print("self.centroids")
            #for i in range(len(self.centroids)):
            #    print(f"x: {self.centroids[i].x}, y: {self.centroids[i].y}, z: {self.centroids[i].z}, centroid id: {self.centroids[i].centroid_id}")

            #print("Previouscentroids")
            #for i in range(len(previousCentroids)):
            #    print(f"x: {previousCentroids[i].x}, y: {previousCentroids[i].y}, z: {previousCentroids[i].z}, centroid id: {previousCentroids[i].centroid_id}")

            if(counter == self.k):
                break

#TODO: implement this method yourself, dont have it as a visualisation in a graph, make it output the necessary data so that someone can visualise it in a graph themselves
    def output(self):
        '''
        Temporary use of ChatGPT code to visualise the algorithm in a graph - NOT MY CODE AND WONT BE USED IN PROJECT
        '''
        # Extract unique centroid_ids
        unique_centroid_ids = sorted(list(set(c.centroid_id for c in self.centroids)))

        # Choose a color map
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(unique_centroid_ids)))

        # Create a mapping from centroid_id to color
        centroid_id_to_color = {cid: color for cid, color in zip(unique_centroid_ids, colors)}

        # Assign colors to datapoints based on their centroid_id
        datapoint_colors = [centroid_id_to_color[dp.centroid_id] for dp in self.datapoints]

        if self.is3D:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot Datapoints
            ax.scatter(
                [dp.x for dp in self.datapoints],
                [dp.y for dp in self.datapoints],
                [dp.z if dp.z is not None else 0 for dp in self.datapoints],
                c=datapoint_colors,
                alpha=0.6,
                label='Datapoints'
            )

            # Plot Centroids
            ax.scatter(
                [c.x for c in self.centroids],
                [c.y for c in self.centroids],
                [c.z if c.z is not None else 0 for c in self.centroids],
                c=[centroid_id_to_color[c.centroid_id] for c in self.centroids],
                s=300,
                marker='X',
                edgecolor='black',
                label='Centroids'
            )

            ax.set_title('K-Means Clustering Visualization (3D)')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.legend()
            plt.show()
        else:
            plt.figure(figsize=(8, 6))

            # Plot Datapoints
            plt.scatter(
                [dp.x for dp in self.datapoints],
                [dp.y for dp in self.datapoints],
                c=datapoint_colors,
                alpha=0.6,
                label='Datapoints'
            )

            # Plot Centroids
            plt.scatter(
                [c.x for c in self.centroids],
                [c.y for c in self.centroids],
                c=[centroid_id_to_color[c.centroid_id] for c in self.centroids],
                s=300,
                marker='X',
                edgecolor='black',
                label='Centroids'
            )

            plt.title('K-Means Clustering Visualization (2D)')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()

    '''
    Private Methods for internal functionality
    '''

    def __squaredEuclideanDistance(self, a: ClusterEntity,b: ClusterEntity):
        '''
            This method calculates the Euclidean Distance (ED) between a data point and a centroid
             in both 2D and 3D.

            The distance is still squared to maximise performance of the algorithm (square root can be
            resource intensive when constantly being performed). The exact value of ED is not necessary
            in this algorithm. When determining the closest centroid to a data point, comparing squared ED
            yields the same outcome as it would with true ED.
        '''
        if(b.z != None):
            #TODO: need new method to calculate ED
            return ((a.x - b.x) **2) + ((a.y - b.y) **2) + ((a.z - b.z) **2)
        else:
            return ((a.x - b.x) ** 2) + ((a.y - b.y) ** 2)


    def __createDataPoints(self, row):
        #create a coordinate object and append it to datapoints
        if(self.is3D):
            dataPoint = ClusterEntity(row.iloc[0], row.iloc[1], row.iloc[2])
        else:
            dataPoint = ClusterEntity(row.iloc[0], row.iloc[1])

        self.datapoints.append(dataPoint)

    def __validateInputs(self, k , df):
        # ensures data has been inputted correctly into algorithm
        if not isinstance(k, int):  # k type check
            raise TypeError(f"Expected k to be int, got {type(k).__name__}")
        if not isinstance(df, pd.DataFrame):  # df type check
            raise TypeError(f"Expected df to be pandas DataFrame, got {type(df).__name__}")
        if len(df.columns) > 3:  # df too large check
            raise ValueError("Input data frame has too many dimensions, ensure data frame has 3 columns maximum")
        if len(df.columns) < 2:  # df too small check
            raise ValueError("Input data frame has too little dimensions, ensure data frame has 2 columns minimum")

        # code adapted from SaturnCloud (2023) https://saturncloud.io/blog/how-to-check-if-a-pandas-dataframe-contains-only-numeric-columns/#:~:text=To%20check%20if%20all%20columns,object%20to%20a%20numeric%20dtype.&text=The%20errors%3D'coerce'%20argument,NaN%20values%20in%20the%20DataFrame.
        # checks data frame is fully numeric (int, float, etc. )
        numeric_columns = df.select_dtypes(include=np.number).columns
        if not len(numeric_columns) == len(df.columns):
            # end of adapted code
            raise ValueError("Data Frame expected to be fully numeric, other data types found")

    def __setupAlgorithm(self):
        #create k number of centroids, using random points in the existing data as starting points
        for i in range(self.k):
            randomDFEntry = self.df.sample(n =1)
            if(self.is3D):
                newCentroid = ClusterEntity(randomDFEntry.iloc[0,0],randomDFEntry.iloc[0,1],randomDFEntry.iloc[0,2], i)
            else:
                newCentroid = ClusterEntity(randomDFEntry.iloc[0, 0], randomDFEntry.iloc[0, 1], None, i)
            self.centroids.append(newCentroid)
        print("Centroids: ")
        for i in range(len(self.centroids)):
            print(f"id: {self.centroids[i].centroid_id}, x: {self.centroids[i].x}, y: {self.centroids[i].y}, z: {self.centroids[i].z}")
        # code adaped from GeeksForGeeks (2024a) https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
        # apply a method to each row in the dataframe to populate datapoints
        self.df.apply(self.__createDataPoints, axis=1)
        # end of adapted code
        print("Data Points: ")
        for i in range(len(self.datapoints)):
            print(f"x: {self.datapoints[i].x}, y: {self.datapoints[i].y}, z: {self.datapoints[i].z}")

        #Add column to associate data entries to centroids
        self.df['Closest_Centroid'] = None