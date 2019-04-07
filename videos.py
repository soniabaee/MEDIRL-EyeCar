#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:17:07 2019

@author: sonia
"""
import pandas as pd

class videos:
    
    def __init__(self):#, group, videoFile):
        """
            insert the value of video for each individuals
        """
        self.allVideos =  ['Day_Rain_High_1','Day_Rain_Town_1','Day_Sunny_High_1','Day_Sunny_Town_1','Night_Rain_High_1',
                           'Night_Rain_Town_1','Night_Rain_Town_2','Night_Sunny_High_1','Night_Sunny_Town_1','Night_Sunny_Town_2',
                           'Day_Rain_High_2','Day_Rain_Town_2','Day_Rain_Town_3','Day_Sunny_High_2','Day_Sunny_Town_3', 'Day_Sunny_Town_4','Night_Rain_High_2',
                           'Nigh_Rain_High_3','Night_Rain_High_4','Night_Rain_Town_3','Night_Sunny_High_2'] ## list of all videos
#        self.slcGroup = group
#        self.videoGroup = self.allVideos[0:10]
        self.videoFile = "/Users/soniabaee/Documents/Projects/EyeCar/eyeCar-Data/Data/InputData/hazardousFrame.csv"
        self.hazardousFrame = {video: {'startFrame':0,'endFrame':0} for video in self.allVideos}
     
    
    def hazardousFrameFun(self):
        """
            from which frame the hazardous object was appeared
        """
        videoInfo = pd.read_csv(self.videoFile)
        
        self.hazardousFrame = {video: {'startFrame':videoInfo.loc[videoInfo['Video'] == video]['startFrame'],
                                       'endFrame':videoInfo.loc[videoInfo['Video'] == video]['endFrame']} for video in self.allVideos}
        
        
        