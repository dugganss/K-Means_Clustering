from random import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from ClusterEntity import ClusterEntity

class KMeans:
    def __init__(self, k: int , df: pd.DataFrame):
        self.__validateInputs(k,df)

        self.centroids = []
        self.datapoints = []
        self.k = k
        self.df = df

        self.__setupAlgorithm()

    '''
    Public Methods for user interaction
    '''
    '''
    TODO: 
    - need to make elbow method find largest change in inertia and use that result
    - change comments across codebase
    - find a way to set manage k so that the user can use elbow method or manually define k 
    (would be preferable if defining k was optional when instantiating and when calling run without k it uses the 
    elbow method and otherwise runs the algo without elbow)
    '''
    def elbow(self):
        results = {}
        for x in range(1, 11):
            kmeans = KMeans(x, self.df)
            results |= kmeans.run()
        kWithLowestInertia = min(results, key=lambda k: results[k][0])
        print("lowest")
        print(results[kWithLowestInertia])
        print("ALl")
        for x in results:
            print(results[x])

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
        results = {}
        MAX_ITERATIONS = 30
        for x in range(MAX_ITERATIONS):
            previousCentroids = copy.deepcopy(self.centroids)
            '''
            This loop is responsible for calculating the squared Euclidean Distance between all data points and all centroids.
            For each data point, it calculates the distance to all centroids and links itself to the centroid closest to it.
            '''
            inertia = 0
            for datapoint in self.datapoints:
                currentCentroidDistances = {}
                for centroid in self.centroids:
                    #create a key value pair, linking the current centroid id to the distance from the current data point
                    squaredDistance = self.__squaredEuclideanDistance(datapoint, centroid)
                    currentCentroidDistances[centroid.centroid_id] = squaredDistance
                    inertia += squaredDistance

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
                    if(datapoint.centroid_id == centroid.centroid_id):
                        associatedDataPointsToCurrentCentroid.append(datapoint)
                if(len(associatedDataPointsToCurrentCentroid) != 0):
                    #sets each dimension of each centroid to the average of the respective associated datapoints' dimensions
                    for i in range(len(centroid.dimensions)):
                        centroid.dimensions[i] = sum(dp.dimensions[i] for dp in associatedDataPointsToCurrentCentroid) / len(associatedDataPointsToCurrentCentroid)

            #print("data points with closest associated centroids:")
            #for i in range(len(self.datapoints)):
            #    print(f"x: {self.datapoints[i].x}, y: {self.datapoints[i].y}, z: {self.datapoints[i].z}, centroid id: {self.datapoints[i].centroid_id}")
            '''
            Convergence Check
            uses numpy operations to count how many previousCentroids dimensions are the same as the current centroids dimensions
            '''
            counter = 0
            for i in range(len(previousCentroids)):
                isSame = previousCentroids[i].dimensions == self.centroids[i].dimensions
                if(np.all(isSame)):
                    counter += 1



            #print("self.centroids")
            #for i in range(len(self.centroids)):
            #    print(f"centroid dimensions: {self.centroids[i].dimensions}, centroid id: {self.centroids[i].centroid_id}")

            #print("Previouscentroids")
            #for i in range(len(previousCentroids)):
            #    print(f"x: {previousCentroids[i].x}, y: {previousCentroids[i].y}, z: {previousCentroids[i].z}, centroid id: {previousCentroids[i].centroid_id}")

            if(counter == self.k):
                results = {self.k: [inertia, self.centroids, self.datapoints]}
                break
        return results

#TODO: implement this method yourself, dont have it as a visualisation in a graph, make it output the necessary data so that someone can visualise it in a graph themselves
    def output(self):
        '''
        Temporary use of ChatGPT code to visualise the algorithm in a graph - NOT MY CODE AND WONT BE USED IN PROJECT
        '''
        # Determine number of dimensions by checking the shape of any datapoint
        num_dims = len(self.datapoints[0].dimensions)

        # 1) If only 2D, do 2D plot
        if num_dims == 2:
            # Prepare color mapping
            unique_centroid_ids = sorted(list(set(c.centroid_id for c in self.centroids)))
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(unique_centroid_ids)))
            centroid_id_to_color = {cid: color for cid, color in zip(unique_centroid_ids, colors)}

            # Assign colors to datapoints based on their centroid_id
            datapoint_colors = [centroid_id_to_color[dp.centroid_id] for dp in self.datapoints]

            # Plot DataPoints
            plt.figure(figsize=(8, 6))
            plt.scatter(
                [dp.dimensions[0] for dp in self.datapoints],
                [dp.dimensions[1] for dp in self.datapoints],
                c=datapoint_colors,
                alpha=0.6,
                label='Datapoints'
            )

            # Plot Centroids
            plt.scatter(
                [c.dimensions[0] for c in self.centroids],
                [c.dimensions[1] for c in self.centroids],
                c=[centroid_id_to_color[c.centroid_id] for c in self.centroids],
                s=300,
                marker='X',
                edgecolor='black',
                label='Centroids'
            )
            plt.title('K-Means Clustering Visualization (2D)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.grid(True)
            plt.show()

        # 2) If 3D, do 3D plot
        elif num_dims == 3:
            # Prepare color mapping
            unique_centroid_ids = sorted(list(set(c.centroid_id for c in self.centroids)))
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(unique_centroid_ids)))
            centroid_id_to_color = {cid: color for cid, color in zip(unique_centroid_ids, colors)}

            # Assign colors to datapoints based on their centroid_id
            datapoint_colors = [centroid_id_to_color[dp.centroid_id] for dp in self.datapoints]

            from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot Datapoints
            ax.scatter(
                [dp.dimensions[0] for dp in self.datapoints],
                [dp.dimensions[1] for dp in self.datapoints],
                [dp.dimensions[2] for dp in self.datapoints],
                c=datapoint_colors,
                alpha=0.6,
                label='Datapoints'
            )

            # Plot Centroids
            ax.scatter(
                [c.dimensions[0] for c in self.centroids],
                [c.dimensions[1] for c in self.centroids],
                [c.dimensions[2] for c in self.centroids],
                c=[centroid_id_to_color[c.centroid_id] for c in self.centroids],
                s=300,
                marker='X',
                edgecolor='black',
                label='Centroids'
            )

            ax.set_title('K-Means Clustering Visualization (3D)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.legend()
            plt.show()

        # 3) Otherwise, n>3 => print or return textual summary
        else:
            print(f"The data has {num_dims} dimensions, which is more than 3.")
            print("Direct visualization not implemented. Below is a summary:")

            # Print centroid coordinates
            for c in self.centroids:
                print(f"\nCentroid ID {c.centroid_id}: {c.dimensions}")

                # Print assigned datapoints for each centroid
                assigned_dps = [dp for dp in self.datapoints if dp.centroid_id == c.centroid_id]
                print(f"  Assigned {len(assigned_dps)} datapoints:")
                for dp in assigned_dps[:5]:  # Print first 5, for brevity
                    print(f"    Datapoint coords: {dp.dimensions}")

            print("\n...End of summary for >3 dimensions.\n")

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
        '''
        for hardcoded implementation
        if(b.z != None):
            #TODO: need new method to calculate ED
            return ((a.x - b.x) **2) + ((a.y - b.y) **2) + ((a.z - b.z) **2)
        else:
            return ((a.x - b.x) ** 2) + ((a.y - b.y) ** 2)
        '''
        difference = a.dimensions - b.dimensions
        return (difference**2).sum()



    def __createDataPoints(self, row):
        #create a coordinate object and append it to datapoints
        '''
        if(self.is3D):
            dataPoint = ClusterEntity(row.iloc[0], row.iloc[1], row.iloc[2])
        else:
            dataPoint = ClusterEntity(row.iloc[0], row.iloc[1])
        '''


        self.datapoints.append(ClusterEntity(row))

    def __validateInputs(self, k , df):
        # ensures data has been inputted correctly into algorithm
        if not isinstance(k, int):  # k type check
            raise TypeError(f"Expected k to be int, got {type(k).__name__}")
        if not isinstance(df, pd.DataFrame):  # df type check
            raise TypeError(f"Expected df to be pandas DataFrame, got {type(df).__name__}")
        #if len(df.columns) > 3:  # df too large check
        #    raise ValueError("Input data frame has too many dimensions, ensure data frame has 3 columns maximum")
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
            randomDFRow = self.df.sample(n =1)
            '''
            this code is for hard coded dimensions
            if(self.is3D):
                newCentroid = ClusterEntity(randomDFEntry.iloc[0,0],randomDFEntry.iloc[0,1],randomDFEntry.iloc[0,2], i)
            else:
                newCentroid = ClusterEntity(randomDFEntry.iloc[0, 0], randomDFEntry.iloc[0, 1], None, i)
            '''
            dimensions = []
            for j in range(randomDFRow.size):
                dimensions.append(randomDFRow.iloc[0,j])
            self.centroids.append(ClusterEntity(dimensions, i))
        #print("Centroids: ")
        #for i in range(len(self.centroids)):
        #        print(f"id: {self.centroids[i].centroid_id}, dimensions: {self.centroids[i].dimensions}")
        # code adaped from GeeksForGeeks (2024a) https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
        # apply a method to each row in the dataframe to populate datapoints
        self.df.apply(self.__createDataPoints, axis=1)
        # end of adapted code
        #print("Data Points: ")
        #for i in range(len(self.datapoints)):
        #    print(f"dimensions: {self.datapoints[i].dimensions}")

        #print(np.mean(self.datapoints[0].dimensions, axis=0))
        #Add column to associate data entries to centroids
        #self.df['Closest_Centroid'] = None