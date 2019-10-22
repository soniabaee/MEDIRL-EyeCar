#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:54:00 2018

@author: soniabaee
"""

from imageai.Detection import VideoObjectDetection
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt


color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}
FILENAME = ""

def forSecond(frame_number, output_arrays, count_arrays, average_count, returned_frame):

    plt.clf()
    plt.show()
    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in average_count:
        counter += 1
        labels.append(eachItem + " = " + str(average_count[eachItem]))
        sizes.append(average_count[eachItem])
        this_colors.append(color_index[eachItem])

    plt.subplot(1, 2, 1)
    plt.title("Second : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)

def forFrame(frame_number, output_array, output_count):
#    print("FOR FRAME " , frame_number)
#    print("Output for each object : ", output_array)
    if len(output_array) != 0:
        print("saved")
        frame_Detail = pd.DataFrame(output_array)
        frame_Detail['frame'] = frame_number
        save_path = os.getcwd() + "/FrameObj" + "/" + FILENAME + "_" + "output_" + str(frame_number) + ".csv"
        frame_Detail.to_csv(save_path)
#    print("Output count for unique objects : ", output_count)
#    print("------------END OF A FRAME --------------")

def objectDection(execution_path, save_path,fileName):
    
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
    detector.loadModel()

    global FILENAME 
    FILENAME = fileName.split('.')[0]

    custom_objects = detector.CustomObjects(car = True, truck = True, bus = True)
    video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects,input_file_path=os.path.join( execution_path, fileName),
                                    output_file_path=os.path.join(save_path, fileName.split(".")[0] + "_detected"),
                                    frames_per_second=30, 
                                    frame_detection_interval = 1 ,
                                    per_frame_function = forFrame,
#                                    per_second_function= forSecond,
                                    minimum_percentage_probability = 79,
#                                    return_detected_frame=True, 
                                    log_progress=True)
    print(video_path)