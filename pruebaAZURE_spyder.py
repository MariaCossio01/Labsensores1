import logging
import azure.functions as func
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json
import pickle
import joblib
import pyodbc 
import pandas as pd


azuredriver = "ODBC Driver 17 for SQL Server"
azurebase = "SQl_proyecto"
usuario = "sensores1_2020-1" 
password = "ingenieriaITM20"
server = "servidor-sensoresone-2020-1.database.windows.net"

connStr = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+azurebase+';UID='+usuario+';PWD='+ password)
cursor = connStr.cursor()
SQL_Script = "SELECT * FROM dbo.BASE_SQL_Final"

df = pd.io.sql.read_sql(SQL_Script,connStr)
connStr.close()
Datos= df.to_numpy()

X=Datos[:,:-1]
Y=Datos[:,-1]

 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
modelo = SVC()
modelo.fit(X_train, Y_train)
predicciones = modelo.predict(X_test)
AccActual = accuracy_score(Y_test,predicciones)
json_response = json.dumps(classification_report(Y_test, predicciones),indent=2)