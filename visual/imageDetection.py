#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:01:30 2019

@author: XXX
"""

from imageai.Detection import ObjectDetection
import os


execution_path = "./medirl-master/videos/crash-video"      
save_path = "./medirl-master/Output"


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "frame357.png"), 
                                             output_image_path=os.path.join(execution_path , "image2new.jpg"), 
                                             minimum_percentage_probability=80)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")



