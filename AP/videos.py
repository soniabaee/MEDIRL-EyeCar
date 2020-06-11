#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   the information about the videos
"""
import pandas as pd
import os

class videos:
    
    def __init__(self):#, group, videoFile):
        """
            insert the value of video for each individuals
        """
        self.allVideos =  [2934487,5592471 ,5996103,9886399,9886402,10528128,
        10528254,10814075,10814077,15396983,15396984,16992777,17726433,22484772,
        22485631,23340980,23362586,23671177,23675224,24523230 ,128888417,26508566,
        116154578,128905745,132361827,132361987,151089859,151089962,151090080] 
#        self.slcGroup = group
#        self.videoGroup = self.allVideos[0:10]

        os.chdir("./medirl-master/Code/")
        self.videoFile = VideoDir
        self.hazardousFrame = {video: {'startFrame':0,'endFrame':0} for video in self.allVideos}
     
    
    def hazardousFrameFun(self):
        """
            from which frame the hazardous object was appeared
        """
        videoInfo = pd.read_csv(self.videoFile)
        
        self.hazardousFrame = {video: {'startFrame':videoInfo.loc[videoInfo['Video'] == video]['startFrame'],
                                       'endFrame':videoInfo.loc[videoInfo['Video'] == video]['endFrame']} for video in self.allVideos}
        
        
        