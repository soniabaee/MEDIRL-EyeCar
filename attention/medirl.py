
"""
    We initialize the basic component of the medirl
"""
import pandas as pd
import numpy as np

from videos import videos
from participants import participants

from objMethod import ObjMethod
import random
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


class medirl:
    
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
            Calculate the score of each video based on participant's info
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
        
        scoreIV = { participant:self.calculateIndScore(participant) for participant in particpants}
        scoreDV = { participant:self.calculateDepScore(participant) for participant in particpants}
       
        self.scoreIV = scoreIV
        self.scoreDV = scoreDV
        
        # objDir = "./medirl-master/Data/"
        # objMethod = ObjMethod(objDir)
        
        # scoreIV = objMethod.load_obj("scoreIV")
        # scoreDV = objMethod.load_obj("scoreDV")
        # self.scoreIV = scoreIV
        # self.scoreDV = scoreDV
        
        self.valueState = { participant:{'scoreIV': scoreIV.get(participant), 'scoreDV': scoreDV.get(participant)} for participant in particpants}
        
        return self.valueState
    
    def calcualteState(self):
        """
            initial the value of state for each participant
        """ 
        state = self.calculateStateValue()
        self.state = state
        objDir = "./medirl-master/Data/"
        objMethod = ObjMethod(objDir)
        objMethod.save_obj(self.state, "state")
    
    def caclulateAction(self,participant):
        """
            this calculates the location of the eye in a grid-space.
            we define 7 different location in a grid    
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
        objDir = "./medirl-master/Data/"
        objMethod = ObjMethod(objDir)      
        objMethod.save_obj(self.action, "action")
    
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
      
        objDir = "./medirl-master/Data/"
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
        
        objDir = "C:/Users/Vishesh/Desktop/Sonia/medirl-master/Data/InputData/"
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
        
        objDir = "./medirl-master/Data/"
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

        
        videos = self.videos.allVideos
        self.videos.hazardousFrameFun()
        hazardousFrame = self.videos.hazardousFrame
        data = []
        particpants = self.participants.allParticipants 
        dataDist = { participant:[] for participant in particpants}
        cnt = 0
        rParam = 0
        nParam = 0
        hParam = 0
        rndVar = 0
        vCnt = 0
        for participant in reward.keys():
            dExp = state[participant]['scoreIV']['driving experience']
            cnt = cnt + 1
            nwDict = dict((key,d[key]) for d in reward[participant] for key in d)
            print(participant)
            for video in videos:
                # print(video)
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
                
                if video in nwDict:
                    
                    distanceLeft =  dict((key,d[key]['DistanceLeft'].iat[0]) for d in state[participant]['scoreDV'][video] for key in d)
                    distanceRight =  dict((key,d[key]['DistanceRight'].iat[0]) for d in state[participant]['scoreDV'][video] for key in d)
                    
                    stFrame = hazardousFrame[video]['startFrame'].iat[0]
                    endFrame = hazardousFrame[video]['endFrame'].iat[0]
                    frames = distanceLeft.keys()
                    distance = []
                    for frame in frames:
                        distance.append(np.mean([distanceLeft[frame], distanceRight[frame]]))
                        dTemp = {'x': frame , 'y': np.mean([distanceLeft[frame], distanceRight[frame]])/10, 
                                 's': np.round(np.mean([distanceLeft[frame], distanceRight[frame]])), 
                                 'color': videos.index(video) + 1, 
                                 'sFrame': stFrame,
                                 'eFrame': endFrame,
                                 'participant': participant }
                        dataDist[participant].append(dTemp)
                   
                    vTemp = {'x': videos.index(video) + 1, 'y': (weight*np.abs(nwDict[video]))/dExp, 's': (np.mean(distance)), 'color': cnt }
                        
                    rndVar =np.mean(distance)*(random.randint(2,3)/2)
                else:
                    vTemp = {'x': videos.index(video) + 1, 'y': (weight*rndVar)/dExp, 's':(rndVar), 'color': cnt }
                data.append(vTemp)
        objDir = "./medirl-master/Data/"
        objMethod = ObjMethod(objDir)
        objMethod.save_obj(dataDist, "dataDist")
        import json
        with open('data.txt', 'w') as outfile:
            json.dump(data, outfile)
        with open('datadist.txt', 'w') as outfile:
            json.dump(dataDist, outfile)
        
        
        
    def distPattern(self):
        """
            each individual has their own pattern of looking at the monitor we
            will check that one here
        """
        
        objDir = "./medirl-master/Data/"
        objMethod = ObjMethod(objDir)
        dataDist = objMethod.load_obj("dataDist")
                
        videos = self.videos.allVideos
        
        
        for video in videos:
            vDist = []
            for participant in dataDist.keys():
                pDist = dataDist[participant]
                vDist.append([item for item in pDist if item["color"] == videos.index(video)+1])
            vDist =  [item for sublist in vDist for item in sublist]
            import json
            with open('data-'+video+'.txt', 'w') as outfile:
                json.dump(vDist, outfile)
        
        
        
    def irlComponent(self):
        """
            save the user's pattern based on each frame in each video
            Should consider two types of pattern 
                1. frame by frame for each user (it is just the sequence of value for the dependent values and the independent values are constant for each video)
                2. vide by video in each group
                3. group by group
        """
        
        objDir = "./medirl-master/medirl/"
        objMethod = ObjMethod(objDir)
        state = objMethod.load_obj("state")
        
        reward = objMethod.load_obj("reward")
        dependentValue = pd.read_csv(self.dependentFile)
        particpants = self.participants.allParticipants
        videos = self.videos.allVideos
        self.videos.hazardousFrameFun()
        hazardousFrame = self.videos.hazardousFrame 
        
        irlAction = {}
        irlState = {}
        for p in particpants:
            for v in videos:
                if v in state[p]['scoreDV'].keys():
                    for f in state[p]['scoreDV'][v][0].keys():
                        if p in irlAction.keys():
                            if v in irlAction[p].keys():
                                if f in irlAction[p][v].keys():
                                    irlAction[p][v][f] = 0 if  np.isnan(state[p]['scoreDV'][v][0][f]['FixationDuration'].iat[0]) == True else 1
                                else:
                                    irlAction[p][v][f] = 0
                                    irlAction[p][v][f] = 0 if  np.isnan(state[p]['scoreDV'][v][0][f]['FixationDuration'].iat[0]) == True else 1
                            else:
                                irlAction[p][v] = {}
                                irlAction[p][v] = {f:0}
                        else:
                            irlAction[p] = {}
                            irlAction[p] = {v:{}}
                            irlAction[p][v] = {f: 0}
                            irlAction[p][v][f] = 0 if  np.isnan(state[p]['scoreDV'][v][0][f]['FixationDuration'].iat[0]) == True else 1
                    fixationDuration =  dict((key,d[key]['FixationDuration'].iat[0]) for d in state[p]['scoreDV'][v] for key in d)
                    gazeX =  dict((key,d[key]['gazeX'].iat[0]) for d in state[p]['scoreDV'][v] for key in d)
                    gazeY =  dict((key,d[key]['gazeY'].iat[0]) for d in state[p]['scoreDV'][v] for key in d)
                    study =  dict((key,d[key]['study'].iat[0]) for d in state[p]['scoreDV'][v] for key in d)
                    distance =  dict((key,(d[key]['DistanceRight'].iat[0]+d[key]['DistanceLeft'].iat[0])/2) for d in state[p]['scoreDV'][v] for key in d)
                    pupil =  dict((key,(d[key]['pupilRight'].iat[0]+d[key]['pupilLeft'].iat[0])/2) for d in state[p]['scoreDV'][v] for key in d)
                    irlState[v] = {}
                    irlState[v] = {'fixationDuration': fixationDuration, 'gazeX': gazeX, 'gazeY': gazeY, 'distance': distance, 'pupil': pupil}
        
        
        objMethod.save_obj(irlAction, "irlAction")
        objMethod.save_obj(irlState, "irlState")



    def generate_expert_trajectories(world, reward, terminal):
        n_trajectories = 200         # the number of "expert" trajectories
        discount = 0.9               # discount for constructing an "expert" policy
        weighting = lambda x: x**50  # down-weight less optimal actions
        start = [0]                  # starting states for the expert

        # compute the value-function
        value = S.value_iteration(world.p_transition, reward, discount)
        
        # create our stochastic policy using the value function
        policy = S.stochastic_policy_from_value(world, value, w=weighting)
        
        # a function that executes our stochastic policy by choosing actions according to it
        policy_exec = T.stochastic_policy_adapter(policy)
        
        # generate trajectories
        tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, start, terminal))
        
        return tjs, policy

    def feature_expectation_from_trajectories(features, trajectories):
        n_states, n_features = features.shape

        fe = np.zeros(n_features)

        for t in trajectories:                  # for each trajectory
            for s in t.states():                # for each state in trajectory
                fe += features[s, :]            # sum-up features

        return fe / len(trajectories)           # average over trajectories

    def initial_probabilities_from_trajectories(n_states, trajectories):
        p = np.zeros(n_states)

        for t in trajectories:                  # for each trajectory
            p[t.transitions()[0][0]] += 1.0     # increment starting state

        return p / len(trajectories)            # normalize

    def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
        n_states, _, n_actions = p_transition.shape
        nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
        
        # Backward Pass
        # 1. initialize at terminal states
        zs = np.zeros(n_states)                             # zs: state partition function
        zs[terminal] = 1.0

        # 2. perform backward pass
        for _ in range(2 * n_states):                       # longest trajectory: n_states
            # reset action values to zero
            za = np.zeros((n_states, n_actions))            # za: action partition function

            # for each state-action pair
            for s_from, a in product(range(n_states), range(n_actions)):

                # sum over s_to
                for s_to in range(n_states):
                    za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]
            
            # sum over all actions
            zs = za.sum(axis=1)

        # 3. compute local action probabilities
        p_action = za / zs[:, None]

        # Forward Pass
        # 4. initialize with starting probability
        d = np.zeros((n_states, 2 * n_states))              # d: state-visitation frequencies
        d[:, 0] = p_initial

        # 5. iterate for N steps
        for t in range(1, 2 * n_states):                    # longest trajectory: n_states
            
            # for all states
            for s_to in range(n_states):
                
                # sum over nonterminal state-action pairs
                for s_from, a in product(nonterminal, range(n_actions)):
                    d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * p_transition[s_from, s_to, a]

        # 6. sum-up frequencies
        return d.sum(axis=1)

    def maxent_irl(p_transition, features, terminal, trajectories, optim, init, eps=1e-4):
        n_states, _, n_actions = p_transition.shape
        _, n_features = features.shape

        # compute feature expectation from trajectories
        e_features = feature_expectation_from_trajectories(features, trajectories)
        
        # compute starting-state probabilities from trajectories
        p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

        # gradient descent optimization
        omega = init(n_features)        # initialize our parameters
        delta = np.inf                  # initialize delta for convergence check

        optim.reset(omega)              # re-start optimizer
        while delta > eps:              # iterate until convergence
            omega_old = omega.copy()

            # compute per-state reward from features
            reward = features.dot(omega)

            # compute gradient of the log-likelihood
            e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward)
            grad = e_features - features.T.dot(e_svf)

            # perform optimization step and compute delta for convergence
            optim.step(grad)
            
            # re-compute detla for convergence check
            delta = np.max(np.abs(omega_old - omega))

        # re-compute per-state reward and return
        return features.dot(omega)

    def setup_mdp():
        # create our world
        world = W.IcyGridWorld(size=7, p_slip=0.2)

        # set up the reward function
        reward = np.zeros(world.n_states)
        reward[-1] = 1.0
        reward[8] = 0.65

        # set up terminal states
        terminal = [24]

        return world, reward, terminal

