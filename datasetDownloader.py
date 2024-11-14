import os
from zipfile import ZipFile
import kagglehub
import shutil
def download():
    # Download latest version
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification", path="WELFake_Dataset.csv", force_download=True)
    print("Path to dataset files:", path)

    cwdPath = os.getcwd()
    shutil.move(path, cwdPath+'\\'+'dataset.zip')
    print("Moved to current working directory")
    #os.rename("WELFake_Dataset.csv", 'WELFake_Dataset.zip')

    zip = ZipFile('dataset.zip', 'r')
    print("Zip file created")
    zip.extractall()
    print("Zip file extracted")
    zip.close()
