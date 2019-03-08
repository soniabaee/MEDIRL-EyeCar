#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:09:56 2019

@author: sonia
"""
import pandas as pd
import numpy as np

from videos import videos
from participants import participants

class eyeCar:
    
    def __init__(self, independentFile, dependentFile, hazardousFile):
        """
            initialize the state with the first video
        """
        self.independentFile = independentFile
        self.dependentFile = dependentFile
        self.hazardousFile = hazardousFile
        
        self.videos = videos()
        
        self.participants = participants()
        self.participants.initalParticipant()
        
        self.state = None
        self.action = None
        self.hazardous = None
        self.reward = None
        
        self.hazardousFrame = self.videos.hazardousFrame
        self.hazardousState = {participant: 0 for participant in self.participants.allParticipants}
        self.scoreIV = {participant: 0 for participant in self.participants.allParticipants}
        self.scoreDV = {participant: 0 for participant in self.participants.allParticipants}
        self.valueState = {participant: 0 for participant in self.participants.allParticipants} 
        self.validate = {participant: 0 for participant in self.participants.allParticipants}
        self.reward = {participant: 0 for participant in self.participants.allParticipants}
        
        self.rewardParam = 0.1
        self.alpha = 0.1
        self.gamma = 1
        self.participants.allParticipants
        
    def calculateIndScore(self, participant):
        """
            calculate the independent variable value by 
            age + gender + environment + weather + day + driving experience
        """
        
        independentValue = pd.read_csv(self.independentFile)
        slcVideos = self.videos.allVideos
        
        iValues = independentValue.loc[(independentValue['ID'] == participant)]# & (independentValue['Video'] in slcVideos).any()]
        
        indValue = {}
        indValue = {video:{'age': iValues[iValues['Video'] == video]['Age'][0], 
                        'gender': iValues[iValues['Video'] == video]['Gender'][0], 
                        'weather': iValues[iValues['Video'] == video]['Weather'][0], 
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
        slcVideos = self.videos.allVideos
        dValues = dependentValue.loc[(dependentValue['participant'] == participant) & (dependentValue['stimulusName'] in slcVideos)]
      
        depValue = {video: {'frame': dValues[dValues['stimulusName'] == video]['stimulusName'], 
                        'gazeX': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['GazeX'], 
                        'gazeY': dValues.loc[(dValues['frame'] == dValues[dValues['Video'] == video]['frame']) & (dValues['stimulusName'] == video)]['GazeY'], 
                        'pupilLeft': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['pupilLeft'], 
                        'pupilRight': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['pupilRight'],
                        'DistanceLeft': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['DistanceLeft'], 
                        'DistanceRight': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['DistanceRight'], 
                        'FixationSeq': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['FixationSeq'], 
                        'FixationDuration': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['FixationDuration'], 
                        'study': dValues.loc[(dValues['frame'] == dValues[dValues['stimulusName'] == video]['frame']) & (dValues['stimulusName'] == video)]['study']} for video in slcVideos}     
        
        
        return depValue;

    def calculateStateValue(self):
        """
            calculate the value of state by scoreIV, scoreDV + which video + place of video on row + group
        """        
        particpants = self.participants.allParticipants
        
        self.scoreIV = { participant:self.calculateIndScore(participant) for participant in particpants}
        self.scoreDV = { participant:self.calculateDepScore(participant) for participant in particpants}
        
        
        self.valueState = { participant:{'scoreIV': self.scoreIV.participant, 'scoreDV': self.scoreDV.participant} for participant in particpants}
    
    def calcualteState(self):
        """
            initial the value of state for each participant
        """ 
        state = self.calculateStateValue()
        self.state = state
    
    def caclulateAction(self,participant):
        """
            this is a boolean value if the agent look at the the hazardous
            it should retun the frame number + duration of hit on that frame
        """
        dependentValue = pd.read_csv(self.dependentFile)
        slcVideos = self.videos.allVideos
        vAction = dependentValue.loc[(dependentValue['participant'] == participant) & (dependentValue['stimulusName'] in slcVideos)]
        
        actionValue = {video: {'FixationDuration': vAction.loc[(vAction['frame'] == vAction[vAction['stimulusName'] == video]['frame']) & (vAction['stimulusName'] == video)]['FixationDuration'], 
                        'fttp': vAction.loc[(vAction['frame'] == vAction[vAction['stimulusName'] == video]['frame']) & (vAction['stimulusName'] == video)]['fttp']} for video in slcVideos}     
        
        return actionValue;
    
    def actionValue(self):
        """
            calculate the value of action for each participant
        """
        particpants = self.participants.allParticipants    
        self.action = { participant:self.caclulateAction(participant) for participant in particpants}
        
    
    def loadHazardous(self, participant):
        """
            this shows the value of participant duration of hazardous situation
        """
        hazardousFile = pd.read_csv(self.hazardousFile)
        slcVideos = self.videos.allVideos
        vHazardous = hazardousFile.loc[(hazardousFile['participant'] == participant) & (hazardousFile['stimulusName'] in slcVideos)]
       
        hazardousValue = {video: {'aoiDuration': vHazardous.loc[vHazardous['stimulusName'] == video]['aoiDuration'], 
                        'ttff': vHazardous.loc[vHazardous['stimulusName'] == video]['ttff'],
                        'timeSpent': vHazardous.loc[vHazardous['stimulusName'] == video]['timeSpent'],
                        'fixationCount': vHazardous.loc[vHazardous['stimulusName'] == video]['fixationCount'],
                        'hitTime': vHazardous.loc[vHazardous['stimulusName'] == video]['hitTime'],
                        'avgFixDuration': vHazardous.loc[vHazardous['stimulusName'] == video]['avgFixDuration'] } for video in slcVideos}     
        
        return hazardousValue;
    
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
        hazardousFrame = self.videos.hazardousFrame
        particpants = self.participants.allParticipants    
        self.hazardous = { participant:self.loadHazardous(participant) for participant in particpants}
        hazardous = self.hazardous
        for participant in action.keys:
            for video in action[participant].keys:
                timeLatency = np.divide(hazardous[participant]['hitTime'], hazardous[participant]['aoiDuration'])
                if hazardous[participant]['timeSpent'] != 0:
                    validate[participant] = {video: 1}
                    rewards[participant] = {videos: np.divide(1,timeLatency)}
                else: 
                    validate[participant] = {video: 0}
                    rewards[participant] = {videos: -1}
                
                
        
        self.reward = rewards
        self.validate = validate
        
        
    
    def pattern(self):
        """
            save the user's pattern based on each frame in each video
            Should consider two types of pattern 
                1. frame by frame for each user (it is just the sequence of value for the dependent values and the independent values are constant for each video)
                2. vide by video in each group
                3. group by group
        """
        state = self.state
        action = self.action
        reward = self.reward
        
        participants = self.participants.allParticipants
        
        fPattern = "test"
        
        
        
        
        
        return uPattern;
