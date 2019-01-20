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


execution_path =  "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos"
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "Day_Rain_High_1.mp4"),
                                output_file_path=os.path.join(execution_path, "Day_Rain_High_1_detected_1"),
                                frames_per_second=29, 
                                per_frame_function = forFrame,
                                log_progress=True)
print(video_path)