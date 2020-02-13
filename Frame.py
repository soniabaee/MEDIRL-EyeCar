#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:16:55 2019

@author: soniabaee
"""

import cv2
import os
import glob


os.chdir("/Users/soniabaee/Documents/Projects/EyeCar/Code/")

VideoDir = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/crash-video"

#videos = glob.glob(VideoDir + '/*.mp4')
#for v in videos:
v = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/Video/Day_Sunny_High_1.mp4_chart_detected.avi"
vidcap = cv2.VideoCapture(v)
success,image = vidcap.read()
pathOut = "Day_Sunny_High_1_Charts"
folder = "Frames"
directory = folder +"/"+ pathOut + "/"
if not os.path.exists(directory):
    os.makedirs(directory)
count = 0
success = True
while success:
    cv2.imwrite(os.path.join(directory, "frame{:d}.png".format(count)), image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1