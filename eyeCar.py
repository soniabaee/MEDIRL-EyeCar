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

from objMethod import ObjMethod

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


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
        print(participant)
        iValues = independentValue.loc[(independentValue['ID'] == participant)]# & (independentValue['Video'] in slcVideos).any()]
        
        indValue = {}
        indValue = {video: {'age': iValues[iValues['Video'] == video]['Age'].iat[0], 
                        'gender': iValues[iValues['Video'] == video]['Gender'].iat[0], 
                        'weather': iValues[iValues['Video'] == video]['Weather'].iat[0], 
                        'day': iValues[iValues['Video'] == video]['Time'].iat[0],
                        'driving experience': iValues[iValues['Video'] == video]['Driving Experience'].iat[0],
                        'posInRow': iValues[iValues['Video'] == video]['posInRow'].iat[0]} for video in slcVideos}                      
        return indValue;

    def calculateDepScore(self, participant):
        """
            in each frame of the video what is the value of
            gaze + pupil size + distance + fixation + fttp + car's speed 
        """
        dependentValue = pd.read_csv(self.dependentFile)
        slcVideos = self.videos.allVideos
        dValues = dependentValue.loc[dependentValue['participant'] == participant]
        depValue = {}
        frameDict = {}
        for video in slcVideos:
            frames = dValues[dValues['stimulusName'] == video]['frame']
            for frame in frames:
                frameDict[frame] = {'frame': frame, 
                                    'gazeX': dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['GazeX'], 
                                    'gazeY':dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['GazeY'], 
                                    'pupilLeft':dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['pupilLeft'], 
                                    'pupilRight': dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['pupilRight'],
                                    'DistanceLeft': dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['DistanceLeft'], 
                                    'DistanceRight':dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['DistanceRight'], 
                                    'FixationSeq': dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['FixationSeq'], 
                                    'FixationDuration': dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['FixationDuration'], 
                                    'study':dValues[(dValues['stimulusName'] == video) & (dValues['frame'] == frame)]['study']}
                if video in depValue.keys():
                    depValue[video].append(frameDict)
                else:
                    depValue[video] = []
                    depValue[video].append(frameDict)
                
                
        return depValue;

    def calculateStateValue(self):
        """
            calculate the value of state by scoreIV, scoreDV + which video + place of video on row + group
        """        
        particpants = self.participants.allParticipants
        
#        scoreIV = { participant:self.calculateIndScore(participant) for participant in particpants}
#        scoreDV = { participant:self.calculateDepScore(participant) for participant in particpants}
#        
#        self.scoreIV = scoreIV
#        self.scoreDV = scoreDV
#        
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)
        
        scoreIV = objMethod.load_obj("scoreIV")
        scoreDV = objMethod.load_obj("scoreDV")
        self.scoreIV = scoreIV
        self.scoreDV = scoreDV
        
        self.valueState = { participant:{'scoreIV': scoreIV.get(participant), 'scoreDV': scoreDV.get(participant)} for participant in particpants}
        
        return self.valueState
    
    def calcualteState(self):
        """
            initial the value of state for each participant
        """ 
        state = self.calculateStateValue()
        self.state = state
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)
        objMethod.save_obj(self.state, "state")
        
#        self.state = objMethod.load_obj("state")
        
       
    
    def caclulateAction(self,participant):
        """
            this is a boolean value if the agent look at the the hazardous
            it should retun the frame number + duration of hit on that frame
        """
        dependentValue = pd.read_csv(self.dependentFile)
        slcVideos = self.videos.allVideos
        vAction = dependentValue.loc[dependentValue['participant'] == participant]
        
        actionValue = {}
        frameDict = {}
        for video in slcVideos:
            frames = vAction[vAction['stimulusName'] == video]['frame']
            for frame in frames:
                frameDict[frame] = {'frame': frame, 
                                    'FixationDuration': vAction[(vAction['stimulusName'] == video) & (vAction['frame'] == frame)]['FixationDuration'], 
                                    'study':vAction[(vAction['stimulusName'] == video) & (vAction['frame'] == frame)]['study']}
                if video in actionValue.keys():
                    actionValue[video].append(frameDict)
                else:
                    actionValue[video] = []
                    actionValue[video].append(frameDict) 
        
        return actionValue;
    
    def actionValue(self):
        """
            calculate the value of action for each participant
        """
        particpants = self.participants.allParticipants    
        self.action = { participant:self.caclulateAction(participant) for participant in particpants}
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)      
        objMethod.save_obj(self.action, "action")
#        self.action = objMethod.load_obj("action")
    
    def loadHazardous(self, participant):
        """
            this shows the value of participant duration of hazardous situation
        """


        hazardousFile = pd.read_csv(self.hazardousFile)
        slcVideos = self.videos.allVideos
        vHazardous = hazardousFile.loc[(hazardousFile['participant'] == participant)]
        hazardousValue = {video: {'aoiDuration': vHazardous.loc[vHazardous['stimulusName'] == video]['AOI Total Duration (ms)'],
                                  'visit': vHazardous.loc[vHazardous['stimulusName'] == video]['Respondent ratio-G'],
                                  'ttff': vHazardous.loc[vHazardous['stimulusName'] == video]['TTFF-F (ms)'],
                                  'timeSpent': vHazardous.loc[vHazardous['stimulusName'] == video]['Time spent-G (ms)'],
                                  'fixationCount': vHazardous.loc[vHazardous['stimulusName'] == video]['Fixations Count'],
                                  'hitTime': vHazardous.loc[vHazardous['stimulusName'] == video]['Hit time-G (ms)'],
                                  'tsG': vHazardous.loc[vHazardous['stimulusName'] == video]['Time spent-G (%)'],
                                  'avgFixDuration': vHazardous.loc[vHazardous['stimulusName'] == video]['Average Fixations Duration (ms)'] } for video in slcVideos}     
        
        return hazardousValue;
    
    def rewardValue(self):
        """
            this is a [0-1] value if the agent look at the hazardous (smaller framenumber) and look at 
            the hazardous in longer time we assign more reward to that participant.
        """
      
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)
        self.action = objMethod.load_obj("action")
        
        videos = self.videos
        action = self.action
        rewards = self.reward
        validate = self.validate
      
#        hazardousFrame = self.videos.hazardousFrame
        particpants = self.participants.allParticipants    
        self.hazardous = { participant:self.loadHazardous(participant) for participant in particpants}
        hazardous = self.hazardous
        for participant in action.keys():
            print(participant)
            for video in action[participant].keys():
                if len(hazardous[participant][video]['visit']) != 0:
                    if hazardous[participant][video]['visit'].iat[0] != 0:
                        timeLatency = hazardous[participant][video]['avgFixDuration'].iat[0] #np.divide(hazardous[participant][video]['ffD'].iat[0], hazardous[participant][video]['aoiDuration'].iat[0])
                        if rewards[participant] == 0:
                            rewards[participant] = []
                            rewards[participant].append({video: timeLatency})
                        else:
                            rewards[participant].append({video: timeLatency})
                    else: 
                        if rewards[participant] == 0:
                            rewards[participant] = []
                            rewards[participant].append({video: -1})
                        else:
                            rewards[participant].append({video: -1})
                else:
                    if rewards[participant] == 0:
                            rewards[participant] = []
                            rewards[participant].append({video: -1})
                    else:
                        rewards[participant].append({video: -1})
                    
                        
        self.reward = rewards
        self.validate = validate
        
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)
        objMethod.save_obj(self.reward, "reward")
    
    def pattern(self):
        """
            save the user's pattern based on each frame in each video
            Should consider two types of pattern 
                1. frame by frame for each user (it is just the sequence of value for the dependent values and the independent values are constant for each video)
                2. vide by video in each group
                3. group by group
        """
        
        objDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
        objMethod = ObjMethod(objDir)
        state = objMethod.load_obj("state")
        action = objMethod.load_obj("action")
        reward = objMethod.load_obj("reward")
        
#        videos = []
#        rewards = []
#        for participant in reward.keys():
#            tmp1 = [list(r.values()) for r in reward[participant]]
#            rewards = [val for sublist in tmp1 for val in sublist]
#            tmp2 = [list(r.keys()) for r in reward[participant]]
#            videos = [val for sublist in tmp2 for val in sublist]
#            rewards = [0 if x==-1 else x for x in rewards]
#            rewards = [-x if x<0 else x for x in rewards]
#            plt.figure(figsize=(10,5))
#            plt.plot(videos, rewards, 'bo')
#            plt.show()
#            data
        import random
        videos = self.videos.allVideos
        data = []
        cnt = 0
        rParam = 0
        nParam = 0
        hParam = 0
        rndVar = 0
        for participant in reward.keys():
            dExp = state[participant]['scoreIV']['Day_Rain_High_1']['driving experience']
            cnt = cnt + 1
            nwDict = dict((key,d[key]) for d in reward[participant] for key in d)
            print(participant)
            for video in videos:
                print(video)
                if "Rain" in video:
                    rParam = 3
                else:
                    rParam = 1
                if "Night" in video:
                    nParam = 2
                else:
                    nParam = 1
                if "High" in video:
                    hParam = 2.5
                else:
                    hParam = 1
                weight = rParam*hParam*nParam
                 #random.randint(1,15)
                if video in nwDict:
                    vTemp = {'x': videos.index(video) + 1, 'y': (weight*np.abs(nwDict[video]))/dExp, 's': (weight*np.round(np.abs(nwDict[video]))*10)/dExp, 'color': cnt }
                    rndVar = np.abs(nwDict[video])*(random.randint(2,5)/2)
                else:
                    vTemp = {'x': videos.index(video) + 1, 'y': (weight*rndVar)/dExp, 's':(weight*rndVar*10)/dExp, 'color': cnt }
                data.append(vTemp)
        import json
        with open('data.txt', 'w') as outfile:
            json.dump(data, outfile)
        fPattern = "test"
        
        
        
        
        
        
