# import necessary libraries
from medirl import medirl
from VFE.VFE import *
from IE.IE import *
from AP.AP import *

import os
import pandas as pd

def main():

	
	directory = "./medirl-master/videos/crash-video"    
    saveDir = "./medirl-master/Output"
    fileNames = os.listdir(directory)
    sub = ".mp4"
    fileNames = [s for s in fileNames if sub in s]
    
    #---------------------------------------------------
	#VFE component
    VFEOutput =[]
    for file in fileNames:
        print(file)
        VFE(directory, saveDir, file)
    
    #---------------------------------------------------
    #IE component
    IEoutput = []
    for file in fileNames:
        print(file)
        IEoutput.append(IE(directory, saveDir, file))

    #---------------------------------------------------
    #AP component
    AP(IEoutput,VFEOutput)
    
    
if __name__ == '__main__':
    main()