#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:58:27 2018

@author: soniabaee
"""

import pandas as pd
import os, glob


def combineCSVFiles(directoryPath):
    csvFiles = glob.glob(directoryPath+'/*.csv')
    combined_csv = pd.concat( [ pd.read_csv(f) for f in csvFiles ] )
    saveDirectory = os.getcwd() + "/combinedData.csv"
    combined_csv.to_csv(saveDirectory)


def main():
    
    directoryPath = os.getcwd()
    combineCSVFiles(directoryPath)
     
    
    

if __name__ == '__main__':
    main()
    
    
import matplotlib.pyplot as plt 
frameLuminosity = pd.DataFrame(frameLuminosityInfo)
mean = frameLuminosity.iloc[0,:]
std = frameLuminosity.iloc[1,:]
    
n, bins, patches = plt.plot(x=mean, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)    