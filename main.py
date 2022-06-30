from flask import Flask,request,render_template
from data_validation import raw_data_validation
from Train_Models import Clustering,Training_models
import data_transformation
import Models
import csv
import pandas as pd
"""app=Flask(__name__)
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/Train", methods=['POST'])"""
def TrainRouteClient():
    path="Training_Batch_Files"
    data_valid=data_transformation.Datavalid(path)
    data_valid.datavalid()
    data=pd.read_csv("Training_final_db/InputFile.csv")
    columns=[i for i in data.columns]
    data_trans=data_transformation.data_tranform(data,columns)
    data=data_trans.trans_data()
    #______________________
    c=Clustering.Cluster(data)
    n=c.Number_clusters(data)
    cluster_data=c.Cluster(data,n)
    #____________________
    #cluster_data=Models.MLmodel.cluster_model(data)
    cluster_data.to_csv("Training_final_db/Cluster_data.csv",index=False)
    data=pd.read_csv("Training_final_db/Cluster_data.csv")
    M=Models.MLmodel(data)
    M.model_save(data)
class Predict_data:
    def __init__(self):
        pass
    def predict_save(self):
        data=pd.read_csv("Training_final_db/Cluster_data.csv")
        m=Models.MLmodel(data)
        m.load_right_model(data)
"""if __name__ == "__main__":
    app.run(debug=True)"""

TrainRouteClient()
Predict=Predict_data()
Predict.predict_save()


