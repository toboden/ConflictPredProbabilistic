import os
from joblib import dump

current_dir = os.getcwd()

from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

from pydrive.drive import GoogleDrive

drive = GoogleDrive(gauth)

var = 1234

# save variables in joblib file
dump(var, 'COLIBtask2_baseline_variables.joblib')


folder_id = '1CNBTHtBOTFXh01WUpP2EF1aEmDi0rSyg'
file1 = drive.CreateFile({'parents':[{u'id': folder_id}]})
file1.SetContentFile('COLIBtask2_baseline_variables.joblib')
file1.Upload()