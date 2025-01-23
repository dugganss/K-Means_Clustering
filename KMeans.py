import pandas as pd
import numpy as np

from ClusterEntity import ClusterEntity

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
        #currently creates a dummy coordinate and a random centroid to test that euclidean distance works
        #randomDFEntry = self.df.sample(n=1)
        #coord = Coordinate(randomDFEntry.iloc[0,0],randomDFEntry.iloc[0,1],randomDFEntry.iloc[0,2])
        #print(f"coord x {coord.x} y {coord.y} z {coord.z}, centroid x {self.centroids[0].x} y {self.centroids[0].y} z {self.centroids[0].z}")
        #print(self.__squaredEuclideanDistance(coord,self.centroids[0]))

        for i in range(len(self.datapoints)):
            currentCentroidDistances = {}
            for j in range(len(self.centroids)):
                #create a key value pair, linking the current centroid id to the distance from the current data point
                currentCentroidDistances[self.centroids[j].centroid_id] = self.__squaredEuclideanDistance(self.datapoints[i], self.centroids[j])

                #print(f" data point: {i} to centroid: {self.centroids[j].centroid_id}: {self.__squaredEuclideanDistance(self.datapoints[i], self.centroids[j])}")
            #set the datapoint's centroid id to the id of the centroid with the shortest distance
            self.datapoints[i].centroid_id = min(currentCentroidDistances, key=lambda distance: currentCentroidDistances[distance])

        print("data points with closest associated centroids:")
        for i in range(len(self.datapoints)):
            print(f"x: {self.datapoints[i].x}, y: {self.datapoints[i].y}, z: {self.datapoints[i].z}, centroid id: {self.datapoints[i].centroid_id}")

        '''
        What needs to be done:
        - iterate through datapoints and find the euclidean distance (ed) to each centroid in centroids
        - set the Closest_Centroid column to the id of the centroid that yields the lowest ed
        - then iterate through all the rows in the df for each associated centroid and find the average ed to its centroid for all the data points linked to that centroid 
        - set the centroids xyz to the average value 
        - have if statement that breaks when previous centroid location barely changes 
        will have to think about how to implement this properly and how you will manage the values that you need
        probably extract some of this to other private methods declared below euclid d method
        '''


    def output(self):
        print("needs implementing")


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