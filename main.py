from KMeans import KMeans
import pandas as pd

#read csv data to pandas dataframe
df = pd.read_csv("wine-clustering.csv")

#redefine dataframe containing columns to be processed by KMeans
df = df[["Alcohol", "Magnesium"]]#, "Flavanoids"]]

#df = pd.DataFrame({
#    'Age': [25, 30, 22],
#    'Salary': [50, 60, 45],
#    'Score': [80, 85, 78]
#})

#create KMeans object, passing k and the dataframe to be processed
kmeans = KMeans(5, df)

kmeans.run()