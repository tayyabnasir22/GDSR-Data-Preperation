import os
from Utilities.PathManager import PathManager
import zipfile
import gdown
import sys

def DownloadAndExtract(gdrive_url, name = "downloaded_file.zip", extract = True):
    os.makedirs(PathManager.GetBasePath(), exist_ok=True)

    zip_path = os.path.join(PathManager.GetBasePath(), name)

    print("Downloading from Google Drive...")
    gdown.download(gdrive_url, zip_path, fuzzy=True)

    print("Download complete.")
    if extract:
        print("Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(PathManager.GetBasePath())

        print("Extraction complete.")

        # Optional: remove zip after extraction
        os.remove(zip_path)
        print("Zip file removed.")

def main():
    # Replaec the drive link with the base dataset files

    # NYU_V2 - Should point to the .mat file, and not the compressed/zip file
    DownloadAndExtract('https://drive.google.com/uc?id=##########', 'nyu_depth_v2_labeled.mat', False)

    # RGBD-D - You need to change this link after getting data from authors as they have not published it, but provide it exclusively on demand
    DownloadAndExtract('https://drive.google.com/uc?id=##########')
    
    # TOFDSR - Currently point to the GDrive link proivded by authors
    #### 1iQEtPBVIA8pmYeSh6YKH93CsJWik1EJz
    DownloadAndExtract('https://drive.google.com/uc?id=##########')
    
if __name__ == '__main__':
    argsLen = len(sys.argv) - 1
    print(argsLen)

    if argsLen < 1:
        print('Setting default path')
        PathManager.BASE_PATH = './'
    else:
        PathManager.BASE_PATH = sys.argv[1]

    print('Base path: ', PathManager.BASE_PATH)
    main()