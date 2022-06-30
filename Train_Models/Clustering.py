import pickle
import pandas as pd
import kneed
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self,data):
        self.data=data
    def Number_clusters(self,data):
        self.data =data
        wcss=[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,11),wcss)
        plt.xlabel("No of clusters")
        plt.ylabel("WCSS")
        plt.title("ELBOW PLOT")
        plt.savefig("Train_Models/k-means_elbow.PNG")
        self.kn=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
        print(self.kn.knee)
        return self.kn.knee
    def Cluster(self,data,no_of_clusters):
        self.kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', random_state=42)
        self.y_means=self.kmeans.fit_predict(data)
        self.data["cluster"]=self.y_means
        with open("Models" + '/' + "cluster" + '.sav', 'wb') as f:
            pickle.dump(self.kmeans, f)
        return self.data
