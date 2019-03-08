# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:14:25 2019

@author: Vishesh
"""

from eyeCar import eyeCar

def main():
    
    dirInd = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/videoPos-demogData.csv"
    dirDep = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/FrameData.csv"
    dirHzrd = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/AOIData.csv"
    
    eyecar = eyeCar(dirInd, dirDep, dirHzrd)
    
    eyecar.calcualteState()
    eyecar.actionValue()
    eyecar.rewardValue()
    
    
    
    
    
if __name__ == '__main__':
    main()