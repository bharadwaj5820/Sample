import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logger import logger
import shutil
from os import listdir
import json
class raw_data_validation:
    def __init__(self,path):
        self.batch_directory=path
        self.Schema_path="Schema_training.json"
    def schema_data(self):
            with open(self.Schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
                LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
                LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
                column_names = dic['ColName']
                NumberofColumns = dic['NumberofColumns']
                print("LengthOfDateStampInFile:{}".format(LengthOfDateStampInFile) + "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile + "\t " + "NumberofColumns:: %s" % NumberofColumns + "\n")
            return LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,column_names
    def manual_regex(self):
        regex= "['cement_strength']+['\_'']+[\d_]+[\d]+\.csv"
        return regex
    def create_good_data_folder(self):
        if os.path.isdir("Training_raw_data/good_rawdata/"):
            pass
        else:
            os.mkdir("Training_raw_data/good_rawdata/")
    def create_bad_data_folder(self):
        if os.path.isdir("Training_raw_data/bad_rawdata/"):
            pass
        else:
            os.mkdir("Training_raw_data/bad_rawdata/")
    def delete_gooddatafolder(self):
        if os.path.isdir("Training_raw_data/good_rawdata/"):
            shutil.rmtree("Training_raw_data/good_rawdata/")
    def delete_baddatafolder(self):
        if os.path.isdir("Training_raw_data/bad_rawdata/"):
            shutil.rmtree("Training_raw_data/bad_rawdata/")
    def validation_file_name(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        self.delete_baddatafolder()
        self.delete_gooddatafolder()
        self.create_bad_data_folder()
        self.create_good_data_folder()
        only_files=[f for f in listdir(self.batch_directory)]
        for f in only_files:
            if re.match(regex,f):
                split_dot=re.split(".csv",f)
                split_dot=re.split("_",split_dot[0])
                if len(split_dot[2])==LengthOfDateStampInFile:
                    if len(split_dot[3])==LengthOfTimeStampInFile:
                        shutil.copy("Training_Batch_Files/"+f,"Training_raw_data/good_rawdata/")
                    else:
                        shutil.copy("Training_Batch_Files/" + f, "Training_raw_data/bad_rawdata/")
                else:
                    shutil.copy("Training_Batch_Files/" + f, "Training_raw_data/bad_rawdata/")
            else:
                shutil.copy("Training_Batch_Files/" + f, "Training_raw_data/bad_rawdata/")
    def columnlength_validation(self,NumberofColumns):
        only_files=[f for f in listdir('Training_raw_data/good_rawdata/')]
        for file in only_files:
            print(file)
            data=pd.read_csv('Training_raw_data/good_rawdata/'+file)
            if data.shape[1]==NumberofColumns:
                pass
            else:
                shutil.move("Training_raw_data/good_rawdata/" + file,"Training_raw_data/bad_rawdata/")
