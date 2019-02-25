#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:09:56 2019

@author: sonia
"""
import pandas as pd

class eyeCar:
    
    def __init__(self, videos, participants, independentFile, dependentFile, state, action, reward):
        """
            initialize the state with the first video
        """
        self.videos = videos
        self.participants = participants
        self.independentFile = independentFile
        self.dependentFile = dependentFile
        self.state = None
        self.action = None
        self.reward = 0
        self.hazardousFrame = self.videos.hazardousFrame
        self.scoreIV = {participant: 0 for participant in self.participants.allParticipants}
        self.scoreDV = {participant: 0 for participant in self.participants.allParticipants}
        self.valueState = {participant: 0 for participant in self.participants.allParticipants} 
        self.validate = {participant: 0 for participant in self.participants.allParticipants}
        self.reward = {participant: 0 for participant in self.participants.allParticipants}
        self.rewardParam = 0.1
        self.alpha = 0.1
        self.gamma = 1
        
        
    def calculateIndScore(self, participant):
        """
            calculate the independent variable value by 
            age + gender + environment + weather + day + driving experience
        """
        
        independentValue = pd.read_csv(self.independentFile)
        slcVideos = self.videos.videoGroup
        
        iValues = independentValue.loc[(independentValue['ID'] == participant) & (independentValue['Video'] in slcVideos)]
        
        indValue = {video:{'age': iValues[iValues['Video'] == video]['Age'][0], 
                        'gender': iValues[iValues['Video'] == video]['Gender'][0], 
                        'weather': iValues[iValues['Video'] == video]['Weather'][0], 
                        'environment': iValues[iValues['Video'] == video]['Environment'][0],
                        'day': iValues[iValues['Video'] == video]['Time'][0],
                        'driving experience': iValues[iValues['Video'] == video]['Driving Experience'][0],
                        'posInRow': iValues[iValues['Video'] == video]['posInRow'][0]} for video in slcVideos}                      
        return indValue;

    def calculateDepScore(self, participant):
        """
            in each frame of the video what is the value of
            gaze + pupil size + distance + fixation + fttp + car's speed
        """
        dependentValue = pd.read_csv(self.dependentFile)
        slcVideos = self.videos.videoGroup
        dValues = dependentValue.loc[(dependentValue['ID'] == participant) & (dependentValue['Video'] in slcVideos)]
        
        depValue = {video: {'frame': dValues[dValues['Video'] == video]['frame'], 
                        'gazeX': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['gazeX'], 
                        'gazeY': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['gazeY'], 
                        'pupilSize': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['pupilSize'], 
                        'distance': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['distance'], 
                        'fixation': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['fixation'], 
                        'fttp': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['fttp'], 
                        'speed': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['Video'] == video)]['speed']} for video in slcVideos}     
        
        
        return depValue;

    def calculateStateValue(self):
        """
            calculate the value of state by scoreIV, scoreDV + which video + place of video on row + group
        """        
        self.scoreIV = { participant:self.calculateIndScore(participant) for participant in particpants}
        self.scoreDV = { participant:self.calculateDepScore(participant) for participant in particpants}
        
        self.valueState = { participant:{'scoreIV': self.scoreIV.participant, 'scoreDV': self.scoreDV.participant} for participant in particpants}
    
    
    def caclulateAction(self,participant):
        """
            this is a boolean value if the agent look at the the hazardous
            it should retun the frame number + duration of hit on that frame
        """
        dependentValue = pd.read_csv(self.dependentFile)
        slcVideos = self.videos.videoGroup
        vAction = dependentValue.loc[(dependentValue['ID'] == participant) & (dependentValue['Video'] in slcVideos)]
        
        actionValue = {video: {'duration': vAction.loc[(vAction['frame'] == vAction[vAction['Video'] == video]['frame']) & (vAction['Video'] == video)]['duration'], 
                        'fttp': vAction.loc[(vAction['frame'] == vAction[vAction['Video'] == video]['frame']) & (vAction['Video'] == video)]['fttp']} for video in slcVideos}     
        
        return actionValue;
    
    def actionValue(self):
        """
            calculate the value of action for each participant
        """
        particpants = self.participants.allParticipants    
        self.action = { participant:self.caclulateAction(participant) for participant in particpants}
        
    
    def rewardValue(self):
        """
            this is a [0-1] value if the agent look at the hazardous (smaller framenumber) and look at 
            the hazardous in longer time we assign more reward to that participant.
        """
        #Erfan:
        #duration of start to end is important if duration is longer the participant has more time to detect but shorter
        #he/she has less time to detect, how shall I involve this in the reward function.?
        videos = self.videos
        action = self.action
        rewards = self.reward
        validate = self.validate
        hazardousFrame = videos.hazardousFrame
        for participant in action.keys:
            for video in action[participant].keys:
                diffStart = action[prticipant][video]['fttp'] - hazardousFrame[video]['start']
                if action[prticipant][video]['fttp'] > hazardousFrame[video]['start'] & action[prticipant][video]['fttp'] > hazardousFrame[video]['end']:
                    validate[participant] = {video: 1}
                    rewards[participant] = {videos: (1/diffStart)}
                else: 
                    validate[participant] = {video: 0}
                    rewards[participant] = {videos: -1}
                
                
        
        self.reward = rewards
        self.validate = validate
        
        
    
    def pattern():
        """
            save the user's pattern based on each frame in each video
        """
        
        return uPattern;
