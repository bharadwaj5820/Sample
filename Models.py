import pandas as pd
import numpy as np
from Train_Models import Clustering,Training_models
from sklearn.model_selection import train_test_split
import os
from File_operations import file_methods
import pickle
from File_operations import file_methods
from Train_Models import Clustering
import pandas as pd

class MLmodel:
    def __init__(self,data):
        self.data=data
        self.cluster= Clustering.Cluster(self.data)
        self.model_train=Training_models.Best_Models()
    def cluster_model(self):
        self.cluster.Number_clusters(self.data)
        self.no_of_clusters=self.cluster.Number_clusters(self.data)
        self.data_cluster=self.cluster.Cluster(self.data,self.no_of_clusters)
        file=file_methods.File_methods()
        #file.Save_model(self.cluster_model,"Cluster_model")
        return self.data_cluster
    def model_save(self,data):
        self.data= data
        #self.data=pd.read_csv(self.data)
        list_clusters=self.data["cluster"].unique()
        print(list_clusters)
        for i in list_clusters:
            Clustering_data=self.data[self.data["cluster"]==i]
            X=Clustering_data.drop(["cluster","Concrete compressive strength(MPa, megapascals) "],axis=1)
            y=Clustering_data["Concrete compressive strength(MPa, megapascals) "]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            best_model_name,best_model=self.model_train.bestmodel(X_train,y_train,X_test,y_test)
            file=file_methods.File_methods()
            file.Save_model(best_model,best_model_name+"_"+str(i))
    def load_right_model(self,data):
        self.data=data
        #self.data=pd.read_csv(self.data)
        no_of_clusters=len(self.data["cluster"].unique())
        for i in range(3):
            data1=data[data["cluster"]==i]
            print(data1.head())
            X = data1.drop(["cluster", "Concrete compressive strength(MPa, megapascals) "], axis=1)
            y = data1["Concrete compressive strength(MPa, megapascals) "]
            f=file_methods.right_model(i)
            e=f.split1()
            dir="Models"
            path=dir+"/"+e+"/"+e+".sav"
            with open(path,"rb") as file:
                model=pickle.load(file)
            predict=model.predict(X)
            inde=data1.index
            data2=pd.DataFrame(list(predict),columns=["Predict"],index=inde)
            print((predict))
            print(data2.head())
            i+=1
            data2.to_csv("Models_new/Data{}.csv".format(i))


        df=pd.concat(map(pd.read_csv,["Models_new/Data1.csv","Models_new/Data2.csv","Models_new/Data3.csv"]),ignore_index=False,axis=0)
        df.sort_values(by=["Unnamed: 0"],ascending=True,inplace=True)
        print(df.head())
        print(df.columns)
        df.to_csv("Models/Output_csv")
        df2=pd.read_csv("Models/Output_csv")
        df3=pd.concat([data,df2],ignore_index=False,axis=1)
        df3.drop(["Unnamed: 0","Unnamed: 0.1"],axis=1)
        return df3.to_csv("Models/Output_csv")