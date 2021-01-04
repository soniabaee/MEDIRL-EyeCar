# import necessary libraries
from visual.visual import *
from driving.driving import *
from tool import fixation
from attention.attention import *

import os

def main():
    directory = "./medirl-master/videos/crash-video"    
    saveDir = "./medirl-master/Output"
    fileNames = os.listdir(directory)
    sub = ".mp4"
    VfileNames = [s for s in fileNames if sub in s]

    sub = ".txt"
    efileNames = [s for s in fileNames if sub in s]
    
    #---------------------------------------------------
	#visual module
    visualOutput =[]
    for file in VfileNames:
        print(file)
        visualOutput = visual(directory, saveDir, file)
    
    #---------------------------------------------------
    #driving module
    drivingOutput = []
    for file in VfileNames:
        print(file)
        drivingOutput = driving(directory, saveDir, file)

    #---------------------------------------------------
    #eye Fixation
    eyeFixationOutput = []
    for file in efileNames:
        print(file)
        eyeFixationOutput = fixation(directory, saveDir, file)

    #---------------------------------------------------
    #attention module
    attention(visualOutput,drivingOutput, eyeFixationOutput)
    
    
if __name__ == '__main__':
    main()