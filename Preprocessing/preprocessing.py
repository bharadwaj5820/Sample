import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data=pd.read_csv("Training_final_db\InputFile.csv")
columns=data.columns
class Preprocess:
    def __init__(self,data,columns):
        self.data=data
        self.columns=columns
    def isnullcolumns(self):
        self.null_columns=[]
        for column in self.columns:
            if self.data[column].isnull().sum()>0:
                self.null_columns.append(column)
        if len(self.null_columns)>0:
            return True
        else:
            return False
    def Knnimpute_missing_data(self,data):
        imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
        self.data=data
        self.new_array=imputer.fit_transform(self.data)
        self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
        return self.new_data
    def duplicate_remove(self,data):
        self.data=data
        self.data=self.data.drop_duplicates(inplace=True,keep="first")
        return self.data
    def normalize_data(self,X):
        self.X=X
        scaler=MinMaxScaler
        self.data=scaler.fit_transform(self.X)
        return self.data
    def train_data_split(self,X,y):
        self.X=X
        self.y=y
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2)
        return self.X_train,self.X_test,self.y_train,self.y_test
    def Corr_threshold(self,data,threshold):
        self.data=data
        self.threshold=threshold
        self.col_corr=set()
        corr_matrix=self.data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])>self.threshold:
                    col_name=corr_matrix.columns[i]
                    self.col_corr.add(col_name)
        return self.col_corr
    def drop_col(self,data,columns):
        self.data=data
        self.columns=columns
        self.data=self.data.drop(columns,axis=1,inplace=True)
        return self.data





