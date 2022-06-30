import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import csv
import os
from os import listdir
import shutil

class DBoperations:
    def __init__(self):
        self.path="Training_final_db/"
        self.goodfilepath="Training_raw_data/good_rawdata"
        self.badfilepath = "Training_raw_data/bad_rawdata"
    def db_connection(self,databasename):
        conn=sqlite3.connect(self.path+databasename+".db")
        return conn

    def createtable_db(self,databasename,column_names):
        conn=self.db_connection(databasename)
        c=conn.cursor()
        c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table'AND name = 'Good_Raw_Data'")
        if c.fetchone()[0]==1:
            conn.close()
        else:
            for key in column_names.keys():
                type = column_names[key]
                try:
                    # cur = cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Good_Raw_Data'".format(dbName=DatabaseName))
                    conn.execute(
                        'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,
                                                                                                 dataType=type))
                except:
                    conn.execute(
                        'CREATE TABLE  Good_Raw_Data ("{column_name}" {dataType})'.format(column_name=key, dataType=type))

    def insertIntoTableGoodData(self,Database):
        conn = self.db_connection(Database)
        goodfilepath= self.goodfilepath
        badfilepath=self.badfilepath
        onlyfiles = [f for f in listdir(goodfilepath)]
        for file in onlyfiles:
            try:
                with open(goodfilepath+'/'+file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in (line[1]):
                            try:
                                conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=(list_)))
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:

                conn.rollback()

                shutil.move(goodfilepath+'/' + file, badfilepath)

                conn.close()

        conn.close()


    def selectingDatafromtableintocsv(self,Database):


        self.fileFromDb = 'Training_final_db/'
        self.fileName = 'InputFile.csv'
        try:
            conn = self.db_connection(Database)
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            #Make the CSV ouput directory
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            # Open CSV file for writing.
            csvFile = csv.writer(open(self.fileFromDb + self.fileName, 'w', newline=''),delimiter=',', lineterminator='\r\n',quoting=csv.QUOTE_ALL, escapechar='\\')

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

        except Exception as e:
            print(e)
