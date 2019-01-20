#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:51:05 2018

@author: soniabaee
"""

import os
from os import walk
import pandas as pd
import numpy as np



def calculateDistribution(eyeData):
    
    
    eyeData.StimulusName.value_counts()
    
    stimuliList = eyeData.StimulusName.value_counts().keys().tolist()
    
    accurateStimuli= []
    for stimuli in stimuliList:
        nEntry = eyeData.StimulusName.value_counts()[stimuli]
        if nEntry > 10000:
            accurateStimuli.append(stimuli)
    
    
    missingDataRate = []
    for stimuli in accurateStimuli:
        tmp = eyeData[eyeData['StimulusName'] == stimuli]
        missingDataRate.append({'stimuli': stimuli, 'missingRate': len(tmp)/len(tmp[tmp['GazeX'] < 0 ])})
        
    print(missingDataRate)
    
    
        
        
            
    
    


    
def main():
    
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/"
    fileName = "eye_002_Luis" + ".csv"
    eyeData = pd.read_csv(directory+fileName)
    calculateDistribution(eyeData)
    

if __name__ == '__main__':
    main()