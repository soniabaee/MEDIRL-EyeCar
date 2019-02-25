#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:17:07 2019

@author: sonia
"""

class videos:
    
    def __init__(self, group, videoFile):
        """
            insert the value of video for each individuals
        """
        
        self.allVideos = [''] ## list of all videos
        self.slcGroup = group
        self.videoGroup = self.allVideos[0:10]
        self.videoFile = videoFile
        self.hazardousFrame = {video: {'start':0,'end':0} for video in self.allVideos}
    
    
    def hazardousFrame(self):
        """
        from which frame the hazardous object was appeared
        """
        videoInfo = pd.read_csv(self.videoFile)
        
        self.hazardousFrame = {video: {'start':videoInfo.loc[videoInfo['Video'] == video]['startFrame'],
                                       'end':videoInfo.loc[videoInfo['Video'] == video]['endFrame']} for video in self.allVideos}
        
        
        