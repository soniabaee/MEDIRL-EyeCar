#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:54:00 2018

@author: soniabaee
"""

from imageai.Detection import VideoObjectDetection
import os
import pandas as pd


def forFrame(frame_number, output_array, output_count):

    frame_Detail = pd.DataFrame(output_array)
    frame_Detail['frame'] = frame_number
    
    save_path = os.getcwd() + "/" + "output_" + str(frame_number) + ".csv"
    frame_Detail.to_csv(save_path)
    
    print("------------END OF A FRAME --------------")

def objectDection(execution_path, fileName):
    
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
    detector.loadModel()
    
    custom_objects = detector.CustomObjects(car=True, truck = True, bus = True)
    video_path = detector.detectCustomObjectsFromVideo(input_file_path=os.path.join( execution_path, fileName),
                                    output_file_path=os.path.join(execution_path, fileName + "_detected"),
                                    frames_per_second=29, 
                                    minimum_percentage_probability = 70,
                                    log_progress=True)
    print(video_path)