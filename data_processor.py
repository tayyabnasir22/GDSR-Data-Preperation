from Utilities.PathManager import PathManager
from Utilities.ProcessingNYUMat import ProcessingNYUMat
from Utilities.ProcessingRGBDD import ProcessingRGBDD
from Utilities.ProcessingTOFDSR import ProcessingTOFDSR
import sys

def main():
    ProcessingNYUMat.GenerateNPYFiles()
    ProcessingRGBDD.GenerateNPYFiles()
    ProcessingTOFDSR.GenerateNPYFiles()
    
    
if __name__ == '__main__':
    argsLen = len(sys.argv) - 1
    print(argsLen)

    if argsLen < 1:
        print('Setting default path')
        PathManager.BASE_PATH = './'
    else:
        PathManager.BASE_PATH = sys.argv[1]
    main()