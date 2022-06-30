from data_validation import raw_data_validation
from Preprocessing import database,preprocessing
from data_validation import *
import pandas as pd

class Datavalid:
    def __init__(self,path):
        self.rawdatavalidation=raw_data_validation.raw_data_validation(path)
        self.database=database.DBoperations()
    def datavalid(self):
        LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns,column_names=self.rawdatavalidation.schema_data()
        regex=self.rawdatavalidation.manual_regex()
        self.rawdatavalidation.validation_file_name(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)
        self.rawdatavalidation.columnlength_validation(NumberofColumns)
        #self.rawdatavalidation.delete_baddatafolder()
        #self.rawdatavalidation.delete_gooddatafolder()
        self.database.createtable_db("Training",column_names)
        self.database.insertIntoTableGoodData('Training')
        self.database.selectingDatafromtableintocsv('Training')
class data_tranform:
    def __init__(self,data,columns):
        self.data=data
        self.columns=columns
        self.preprocess=preprocessing.Preprocess(self.data,self.columns)
    def trans_data(self):
        while self.preprocess.isnullcolumns():
            self.data=self.preprocess.Knnimpute_missing_data(self.data)
        self.preprocess.duplicate_remove(self.data)
        self.col_remove=self.preprocess.Corr_threshold(self.data,0.9)
        for i in self.col_remove:
            self.preprocess.drop_col(self.data,i)
        return self.data

