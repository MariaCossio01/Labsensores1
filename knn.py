import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
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

n_neighbors = 4
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
Model = KNeighborsClassifier(n_neighbors)
Model.fit(X_train,Y_train.ravel())
Y_es = Model.predict(X_test)
Y_es = np.expand_dims(Y_es,axis=1)
AccActual = accuracy_score(Y_test,Y_es)

json_response = json.dumps(classification_report(Y_test, Y_es),indent=2)
