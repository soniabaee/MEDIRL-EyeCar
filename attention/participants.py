#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	the list of participants and their corresponding videos
"""

class participants:
    
    def __init__(self):
        """
            insert the value of video for each individuals
        """
        
        self.allParticipants = [''] ## list of all participant
        self.participantPerformance = {participant: 0 for participant in self.allParticipants}
        
        
    def initalParticipant(self):
        
        self.allParticipants = ['ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9', 'ID10',
        						'ID11', 'ID12', 'ID13', 'ID14', 'ID15', 'ID16', 'ID17', 'ID18', 'ID19', 'ID20']
        
        