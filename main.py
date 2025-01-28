from KMeans import KMeans
import pandas as pd
import numpy as np

#read csv data to pandas dataframe
df = pd.read_csv("wine-clustering.csv")

#redefine dataframe containing columns to be processed by KMeans
df = df[[ "Alcohol", "Flavanoids", "Color_Intensity"]]

'''
When running the algorithm with columns of varying scale (e.g x = 0-10 and y = 0-1000),
the euclidean distance calculation becomes skewed which when represented in a graph, 
makes it look like datapoints are grouped to the incorrect centroid when in fact, it may be 
closer mathematically to the other node. To combat this, the scales should be normalised 
between dimensions, this should be part of the preprocessing for more accurate grouping 
and this is not a fault in the algorithm. (talk about this in your report) 
'''

#df = pd.DataFrame({
#    'Age': [25, 30, 22],
#    'Salary': [50, 60, 45],
#    'Score': [80, 85, 78]
#})

#create KMeans object, passing the dataframe and optionally a value for K (will be automatically determined otherwise)
kmeans = KMeans(df)

#run the algorithm, saving the results to a variable
results = kmeans.run()

#optionally output the results in a readable format
kmeans.outputToConsole()
print(results)

#kmeans.test()