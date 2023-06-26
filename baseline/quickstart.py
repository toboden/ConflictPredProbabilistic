import os
from joblib import dump
import pyarrow.parquet as pq

current_dir = os.getcwd()

from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

from pydrive.drive import GoogleDrive

drive = GoogleDrive(gauth)

data_folder_id = '1RigGnEyyNGnO_SPBSc_RwO9jjdbnPTAV'
result_folder_id = '1CNBTHtBOTFXh01WUpP2EF1aEmDi0rSyg'

## upload file---------------------------
""" var = 1234

# save variables in joblib file
dump(var, 'COLIBtask2_baseline_variables.joblib')


file1 = drive.CreateFile({'parents':[{u'id': result_folder_id}]})
file1.SetContentFile('COLIBtask2_baseline_variables.joblib')
file1.Upload() """

## import file-----------------------------
# Liste der Dateien im Ordner
file_list = drive.ListFile({'q': f"'{data_folder_id}' in parents and trashed=false"}).GetList()

# Beispiel: Herunterladen der ersten Datei im Ordner

""" file = file_list[0]
file.GetContentFile(file['title']) """

""" # Öffnen der Parquet-Datei und Speichern der Daten in einer neuen Variable
parquet_file = pq.ParquetFile(file['title'])
loaded_data = parquet_file.read().to_pandas()

print(loaded_data.loc[:,'ged_sb']) """





# create the feature- and actuals-data list
# set the feature and actuals year lists
feature_years = ['2017','2018','2019','2020']
actual_years = ['2018','2019','2020','2021']

actuals_df_list = []
features_df_list = []

# store data in lists
for i in range(len(feature_years)):

    feature_title = 'cm_features_to_oct' + feature_years[i] + '.parquet'

    for file in file_list:
        if file['title'] == feature_title:
            file.GetContentFile(file['title'])
            parquet_file = pq.ParquetFile(file['title'])
            #loaded_data[file['title']] = parquet_file.read().to_pandas()

            features_df_list.append({'year':feature_years[i], 'data':parquet_file.read().to_pandas()})

        actual_title = 'cm_actuals_' + actual_years[i] + '.parquet'

        if file['title'] == actual_title:
            file.GetContentFile(file['title'])
            parquet_file = pq.ParquetFile(file['title'])
            #loaded_data[file['title']] = parquet_file.read().to_pandas()

            actuals_df_list.append({'year':actual_years[i], 'data':parquet_file.read().to_pandas()})


print(actuals_df_list[0]['data'].loc[:,'ged_sb'])