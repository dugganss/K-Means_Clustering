from KMeans import KMeans
import pandas as pd

"""
Example of running the Algorithm:

Pre-processing and running steps:
- import pandas
- read csv file to pd dataframe
- Select the columns for processing (create new df)
- Clean the data (remove rows containing na values)
- scale data if needed (when columns have varying ranges of values it can skew grouping because things can
be further away than they seem)
- create K-Means object passing in dataframe and optionally a value for k if you dont want it to be manually inferred
- call the run method on the K-Means object 

There are a few options for managing the output:
1: Can get a list of all the values returned from run() (not recommended)
2: Can call the member variables directly from the object after run() has been called which are as follows:
 - inertia
 - k
 - iterations 
 - list of centroids (ClusterEntity objects) 
 - list of datapoints (Datapoint objects)
3: Can visualise all the results in a console output by calling outputToConsole() after run()
4: Can also get 2 dataframes, the first being a copy of the input with the additional columns 'Associated_Centroid'
and 'Silhouette_Coefficient' for each row. The second being a dataframe of the inferred centroids containing all the 
columns from the original dataframe as well as an additional column 'Centroid_No' which associates to the numbers
contained within 'Associated_Centroid' in the first dataframe. These can be obtained from calling getDataframes() and 
indexing the result [0] or [1].
"""

#read csv data to pandas dataframe
df = pd.read_csv("wine-clustering.csv")

#redefine dataframe containing columns to be processed by KMeans
df = df[[ "Alcohol", "Flavanoids", "Magnesium"]]

#scale the columns to be between 0-1, this only works where max != min
min = df.min()
max = df.max()
scaledDF = (df - min) / (max - min)

#create KMeans object, passing the dataframe and optionally a value for K (will be automatically determined otherwise)
kmeans = KMeans(scaledDF)
"""
e.g. with k defined: 
kmeansWithK = KMeans(scaledDF, 3)
"""
#run the algorithm, optionally saving the results to a variable
results = kmeans.run()

# Managing the results:
#optionally output the results in a readable format
kmeans.outputToConsole()

#call member variables directly from object
k = kmeans.k
inertia = kmeans.inertia
centroids = kmeans.centroids
datapoints = kmeans.datapoints
iterations= kmeans.iterations

#retrieve dataframes
datapointsDF = kmeans.getDataframes()[0]
centroidsDF = kmeans.getDataframes()[1]
