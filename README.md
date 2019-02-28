# eyeCar

## paper
- https://www.hfes.org/blogs/lindsey-kim/2019/01/16/call-for-papers-machine-learning-artificial-intell
## to-do
- should add speed of car for each video during the recording
- the density value of the video is not captured and should be added in the new data frame
**paper related to reinforcement learning in driving**
1. How to define the score for each individual 
    https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
    https://machinelearningmastery.com/feature-selection-machine-learning-python/
    1. one approach: based on the importance of each feature we can define the score by weightening of them
2. The dynamic score should have the determined defintion:
    1. challenge is that what does the gaze, fixation, fixation duration mean should we consider in whole video or during the accident we should check or we should have different levels for that as porpotional. For example, the gaze value during whole video and gaze value during the (+/-6) of the accident. 
3. different sequence of video and effect of that for each individual, we should have position of video on that sequence for that participant.
4. action: is a probability value as looking (+/-6s) at hazardous object or not or change the distance to the monitor (+/-6) of the accident. 
5. Reward: how to define the reward is based on the action and state if the action value probability of detection is higher we would assing more rewards to the participant and update individual state based on the reward. 
6. what is the pattern of each individual based on the state, action, and reward
7. what is the most interesting videos for each individual, what is the meaning of the interest by dependent and independent value
## Idea
- matching realtime video with eye tracking data _(portable eye tracker)_ to see when the participant attention is off-road.
### Title
- Automatic Hazard Recognition by Physiological Data (forward collision avoidance)
### Hyptothesis

- Participant:
    - Independent variable score (score_{I_j}): this is a constant value during watching different videos 
    - Dependent variable score (score_{D_ij}): this is a dynamic value during watching different videos, each video has their own value for eye tracker data, and heart rate for each individual participant

- Video:
    - type of accident is unique
    - we have two different sequences of video to show
        - v1, ..., v21
        - v'1, ..., v'21
- Reinforcement learning framework
    - state: (score_{D_ij}) + (score_{D_ij}) + video's position in sequence of that group
    - action: did the participant look at the hazardous or not (0-1)
    - Reward: the time and duration of looking at hazardous shows the score of detection if the it is close to 1 they get more reward, otherwise get less reward

- we can find the personal pattern of detection for each individual in two different sequence of videos
- the most interesting video would be determined by state, action, reward of participant to see the amount of relationship between independent and dependent variable for each individual


## Variables
    - independent variables: Enivronment, Weather, Density (number of cars in the road), gender, Day time (night, day), Driving experience
    - dependent variables: Eye tracker (Gaze, pupil size, distance, fixation, fttp), HR (HRV, NN50, RMSD), EEG (VEP), car's speed

## to-do list
- [ ] **Driving Framework**
- [x] Create a github page
- [x] Push my commits
- [x] Finish video lighting histogram
- [ ] Check the SHRP2 data with our current data
- [ ] Restore the csv file in repository



## Weekly meeting
Meeting on every Tuesday at 4:45-5:45 pm.

## Problem definition: 
For edititng the paper check this [link](https://www.overleaf.com/7815968227xqkmcmwpzmzv)


Recent studies found that a strong correlation exists between drivers' glance pattern, situation awareness, and hazard detection performance. This study, to the best of authors knowledge, is the first attempt to integrate independent physiological data in a hierarchical mixture of expert models to create a driving assistance systems. The current study presents a preliminary stage in sight of designing a driving assistance system which can constantly monitor the level of attention of a driver in various traffic environments. 
This study will be conducted with \textbf{12?} college students equally balanced between male and female, between the ages of 18 and 35. Participant' physiological data will be monitored using synchronized eye-tracker (Tobii pro X3-120), face recognition (Affectiva), and electroencephalography (Emotiv epoc, 14 channels). Eye-tracker, face recognition, and EEG will be used to respectively capture the gaze and attention level,  facial reactions before/during accidents' occurrence, mental workload, and engagement.
In order to achieve the most effective prospect of drivers, Naturalistic driving study videos for the Second Strategic Highway Research Program (SHRP 2) of Virginia Tech Transportation Institute (VTTI) will be utilized. Semantic segmentation algorithm(s) will be applied on the videos to address the role of the driver's performance and behavior in traffic safety. The most dangerous segments are objects, human, animal, etc. where the participants will be required to carefully search for. Four-Weather conditions ( day, night, sunny, and rainy) road environments ( urban, and freeway)-by-twenty level (observation times) mixed design will be used. All participants will be randomly assigned to each condition. The gender balance will be considered to be evenly distributed between the conditions.
Participants will be required to monitor the videos while their visual behavior, EEG and facial expression will be recorded. Each unique video contains 30 seconds of one crash or near-crash incident with various hazardous objects in the scene. On the one hand, participants will be hypothesized to look for the object(s) and potentially prevent crashes as a normal driver does. On the other hand, an algorithm (has already been designed) will automatically identify hazardous objects in the driver's view. Mapping of both sides adjunct to the participants' physiological data will provide an insight of detection variance of machine and human.
Then a modular neural network architecture, a hierarchical mixture of expert models, will be used in this study. This preliminary study will be the first stage towards creating a safer alternative driving assistance to alert driver once its detection varies from the driver. If the driver fails to perceive a threat, the assistance system triggers a warning. The result from this paper could have an application in industry and academic sectors.


## Data
You can accesss to the data in this [folder](https://drive.google.com/drive/folders/1G-3t3T8QLLeO6fdwetbF7AnjliQA6uDm?usp=sharing)


### Type of SHRP2 dataset
**Note**: we do not need all of them we should choose between that are related to our main goal.

**Vehicle**
- Classification
- Site name
- integrated cell phone
- Controls location
- Factory navigation
- Navigation display location
Trip:
- Start and end UTC: 
    - hour, month,
    - Day of week
- Duration
- Max speed
- Mean speed
- Time moving
- Time not moving
- Brake activation
- Lane tracker right/left side
- Trip distance
- Turn signal available 
- Traction control
- Seatbelt usage percentage
- Vehicle network supports light activation
- Light usage percentage
- Time/distance 0-10/…/70-80/>80
- Alcohol flag
- Cell phone flag
- Urb Frwy
- Usb Frwy < 4 las
- Usb 2 ln
- Speed limit


**Participant**
    - gender
    - Age 
    - Drive mileage last year
    - Participant receive license
    - Average annual mileage
    - Years driving
    - Training
    - Number violations
    - Number of crashes
    - Crash 1 severity
    - Crash 1 fault
    - Crash 2 severity
    - Crash 2 fault
    - Insurance status
    - Vision condition
    - Vision correction
    - Driving vision correction
    - Hearing condition
    - Brain condition
    - Nervous system and sleep condition
    - Age related condition
    - Gave up driving
    - ADHD
        - Easily distracted
        - Difficulty organizing
        - Loses object
        - Quick screen - difficulty waiting turn
        - Feels restless
        - Difficulty enjoying leisure activities
    - Driving
        - Night driving
        - Yellow lights
        - Green arrows
        - Emergency vehicles
        - Dimming lights
        - Merge signs
        - Curve signs
        - Police officer
        - Right of way
        - traffic control
        - Yellow lines
        - Blind spot
        - Drowsiness
        - City driving
        - Light changes
        - Number of correct (out of 19) ???
    - Risk perception
    - Risk taking
    - Sensation seeking
    - Driver behavior
    - Visual and cognitive tests
    - Sleep habits questionnaire
    - clock drawing 
**Event**
    - Severity
    - Start
    - Reaction start
    - Impact and proximity time 
    - Maneuver
    - Precipitating 
    - Vehicle 1,2,3
    - Event nature
    - Incident type
    - Crash severity
    - Driver behavior
    - Driver impairment
    - Front/end seat passenger
    - Hands on the wheel
    - Driver seatbelt use
    - Vehicle contributing factors
    - Infrastructure
    - Visual obstruction
    - Lighting 
    - Weather

### the eventl list of this dataset:
- event ID  event type
- 2834107   light crash
- 2880462   near crash
- 2880464   near crash
- 2934487   crash
- 5592007   crash
- 5592471   crash
- 5996103   crash
- 9886399   crash
- 9886402   crash
- 9886406   crash
- 10528092  near crash
- 10528128  crash
- 10528253  crash
- 10528254  crash
- 10597916  harsh brake
- 10814075  crash
- 10814076  crash
- 10814077  crash
- 10858068  crash
- 10858435  swerve/brake to avoid animal
- 15396983  crash
- 15396984  crash
- 15396985  crash
- 16992777  crash
- 16992786  near crash
- 17726433  crash
- 17749339  near-crash
- 17750281  harsh brake
- 22484772  crash
- 22485021  crash
- 22485631  crash
- 23340980  crash
- 23341336  crash
- 23362586  crash
- 23671177  crash
- 23675202  crash
- 23675224  crash
- 24117980  near crash
- 24516702  crash
- 24523230  crash
- 26235291  crash
- 26508566  additional baseline
- 116154578 Sample Baseline
- 128888417 crash
- 128905745 Sample Baseline
- 132361827 Sample Baseline
- 132361987 Sample Baseline
- 151089859 Sample Baseline
- 151089962 Sample Baseline
- 151090080 Sample Baseline
