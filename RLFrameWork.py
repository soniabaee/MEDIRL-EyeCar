#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:34:06 2019

@author: soniabaee
"""

import pandas as pd
import numpy as np


def calculateIndScore():
    """"
        calculate the independent variable value by 
        age + gender + environment + weather + day + driving experience
    """"
    
    return scoreIV;
    
def calculateDepScore():
    """
        in each frame of the video what is the value of
        gaze + pupil size + distance + fixation + fttp + car's speed
    """
    
    return scoreDV;

def stateValue():
    """
        calculate the value of state by scoreIV, scoreDV + which video + place of video on row + group
    """
    
    return state;

def actionValue():
    """
        this is a boolean value if the agent look at the the hazardous
        it should retun the frame number + duration of hit on that frame
    """
    return action,frame,duration;

def rewardValue():
    """
        this is a [0-1] value if the agent look at the hazardous (smaller framenumber) and look at 
        the hazardous in longer time we assign more reward to that participant.
    """
    return rewards;

def pattern():
    """
        save the user's pattern based on each frame in each video
    """
    
    return uPattern;

def main():
    

if __name__ == '__main__':
    main()