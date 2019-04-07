# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:14:25 2019

@author: Sonia
"""

from eyeCar import eyeCar

def main():
    
    dirInd = "/Users/soniabaee/Documents/Projects/EyeCar/eyeCar-Data/Data/InputData/videoPos-demogData.csv"
    dirDep = "/Users/soniabaee/Documents/Projects/EyeCar/eyeCar-Data/Data/InputData/FrameData.csv"
    dirHzrd = "/Users/soniabaee/Documents/Projects/EyeCar/eyeCar-Data/Data/InputData/AOIData.csv"
    
    eyecar = eyeCar(dirInd, dirDep, dirHzrd)
    
    ## inital the reward, action and state for each participant during the study
#    eyecar.calcualteState()
#    eyecar.actionValue()
#    eyecar.rewardValue()
#    eyecar.pattern()
#    eyecar.distPattern()
    eyecar.irlComponent()
    
    
    
    
    
if __name__ == '__main__':
    main()