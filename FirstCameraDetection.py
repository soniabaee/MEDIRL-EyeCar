#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:30:12 2018

@author: sonia
"""

from imageai.Detection import VideoObjectDetection
import os
import cv2


execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()


video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                             output_file_path=os.path.join(execution_path, "camera_detected_1"),
                                             frames_per_second=29, log_progress=True)

print(video_path)