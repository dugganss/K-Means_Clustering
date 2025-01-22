import matplotlib.pyplot as plt
import pandas as pd

#This code draws a 3d scatter plot to visualise the data (to see what columns create good clustering)

df = pd.read_csv("wine-clustering.csv")

#These columns seem to create a decent amount of defined clusters (will be good to develop with)
columns = ["Alcohol", "Magnesium", "Flavanoids"]

plotData = df[columns]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')

x = df[columns[0]]
y = df[columns[1]]
z = df[columns[2]]

scatter = ax.scatter(x,y,z,c='blue',marker='o',alpha=0.6, edgecolors='w', s = 100)

ax.set_xlabel(columns[0], fontsize=12)
ax.set_ylabel(columns[1], fontsize=12)
ax.set_zlabel(columns[2], fontsize=12)

ax.view_init(elev= 20, azim=30)

plt.show()
