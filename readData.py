#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:28:07 2018

@author: soniabaee
"""


import os
from os import walk
import pandas as pd
import numpy as np


def loadData():
    """ Load data """
        
    directory = os.getcwd()
    studyGroup = "Study_1_Group_1"
    fileName = "002_Luis" 
    AOIFile = "AOI_analysis-MovingAOIResult"
#    mainData = directory + "/" + studyGroup + "/Sensor Data/" + fileName + ".xlsx"
    AOIData = directory + "/Data/" + studyGroup + "/AOI/" + AOIFile + ".xlsx"
    
    dt = pd.ExcelFile(AOIData) 
    sheetNames = dt.sheet_names
    df = dt.parse(sheetNames[0])
    data = df.loc[14:df.shape[0]].values
    data = pd.DataFrame(data=data[1:,0:], columns=data[0,0:])  
    crashCause = data[(data['AOI-Name'] == 'Crash_cause') | 
                        (data['AOI-Name'] == 'crash_cause') | 
                        (data['AOI-Name'] == 'Crash_Cause')]
    
    saveFile = directory + fileName + ".csv"
    data.to_csv(saveFile)
    
    print("sonia")
    
    
def main():
    loadData()
    

if __name__ == '__main__':
    main()
    