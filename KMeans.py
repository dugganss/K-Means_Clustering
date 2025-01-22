import pandas as pd
import numpy as np

from Centroid import Centroid
from Coordinate import Coordinate


class KMeans:
    centroids = []
    def __init__(self, k: int , df: pd.DataFrame):
        #ensures data has been inputted correctly into algorithm
        if not isinstance(k, int):  #k type check
            raise TypeError(f"Expected k to be int, got {type(k).__name__}")
        if not isinstance(df, pd.DataFrame):    #df type check
            raise TypeError(f"Expected df to be pandas DataFrame, got {type(df).__name__}")
        if len(df.columns) > 3:     #df too large check
            raise ValueError("Input data frame has too many dimensions, ensure data frame has 3 columns maximum")
        if len(df.columns) < 2:     #df too small check
            raise ValueError("Input data frame has too little dimensions, ensure data frame has 2 columns minimum")

        #code adapted from SaturnCloud (2023) https://saturncloud.io/blog/how-to-check-if-a-pandas-dataframe-contains-only-numeric-columns/#:~:text=To%20check%20if%20all%20columns,object%20to%20a%20numeric%20dtype.&text=The%20errors%3D'coerce'%20argument,NaN%20values%20in%20the%20DataFrame.
        #checks data frame is fully numeric (int, float, etc. )
        numeric_columns = df.select_dtypes(include=np.number).columns
        if not len(numeric_columns) == len(df.columns):
        #end of adapted code
            raise ValueError("Data Frame expected to be fully numeric, other data types found")

        self.k = k
        self.df = df

    def setupAlgorithm(self):
        #create k number of centroids, using random points in the existing data as starting points
        for i in range(self.k):
            randomDFEntry = self.df.sample(n =1)
            if(len(self.df.columns) == 3):
                newCentroid = Centroid(i,randomDFEntry.iloc[0,0],randomDFEntry.iloc[0,1],randomDFEntry.iloc[0,2])
            else:
                newCentroid = Centroid(i,randomDFEntry.iloc[0, 0], randomDFEntry.iloc[0, 1])
            self.centroids.append(newCentroid)

        #Add column to associate data entries to centroids
        self.df['Closest_Centroid'] = None

    def run(self):
        #currently creates a dummy coordinate and a random centroid to test that euclidean distance works
        randomDFEntry = self.df.sample(n=1)
        coord = Coordinate(randomDFEntry.iloc[0,0],randomDFEntry.iloc[0,1],randomDFEntry.iloc[0,2])
        print(self.__squaredEuclideanDistance(coord,self.centroids[0]))

        '''
        What needs to be done:
        - iterate through each row in the dataframe
        - create a coordinate object at each iteration and calculate euclidean distance (ed) to each centroid
        - set the Closest_Centroid column to the id of the centroid that yields the lowest ed
        - then iterate through all the rows in the df for each associated centroid and find the average ed to its centroid for all the data points linked to that centroid 
        - set the centroids xyz to the average value 
        - have if statement that breaks when previous centroid location barely changes 
        will have to think about how to implement this properly and how you will manage the values that you need
        probably extract some of this to other private methods declared below euclid d method
        '''


    def output(self):
        print("needs implementing")


    # The methods defined below this point are private (internal functionality)
    '''
    The below method calculates the euclidean distance (ED) between a data point and a centroid
     in both 2D and 3D.
     
    The distance is still squared to maximise performance of the algorithm (square root can be
    resource intensive when constantly being performed). The exact value of ED is not necessary 
    in this algorithm. When determining the closest centroid to a data point, comparing squared ED 
    yields the same outcome as it would with true ED.
    '''
    def __squaredEuclideanDistance(self, a: Coordinate,b: Centroid):
        if(b.z != None):
            return ((a.x - b.x) **2) + ((a.y - b.y) **2) + ((a.z - b.z) **2)
        else:
            return ((a.x - b.x) ** 2) + ((a.y - b.y) ** 2)