#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:34:12 2019

@author: sonia
"""

class participants:
    
    def __init__(self):
        """
            insert the value of video for each individuals
        """
        
        self.allParticipants = [''] ## list of all videos
        self.participantPerformance = {participant: 0 for participant in self.allParticipants}
        
        
    def initalParticipant(self):
        
        self.allParticipants = ['Erfan', 'Anna', 'Luis', 'Inki', 'Daniel', 'Victor', 'Janie']
        
        