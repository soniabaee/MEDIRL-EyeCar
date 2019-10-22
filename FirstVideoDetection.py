#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:21:50 2018

@author: sonia
"""

from imageai.Detection import VideoObjectDetection
import os
import pandas as pd

execute_path = os.getcwd()




def forFram(frame_number, output_array, output_count):
#    print("For frame", frame_number)
#    print("output for each object: ", output_array)
#    print("output count for unique object:", output_count)
    
    save_arr = []
    for i in range(len(output_array)):
        save_obj = {}
        save_obj = {"frame": frame_number,"box_points": output_array[i]["box_points"],"name":output_array[i]["name"],"percentage_prob":output_array[i]["percentage_probability"]}
        save_arr.append(save_obj)
    pdObj = pd.DataFrame(save_arr)
    pdObj.to_csv("/Users/sonia/Desktop/Projects/output_"+str(frame_number)+".csv")
    print("----------end of frame--------------")
    
    
def forSeconds(second_number, output_array, count_arrays, average_output_count):
    print("second:", second_number)
#    print("array for the outputs of each frame", output_array)
#    print("array for output count for unique objects in each frame: ", count_arrays)
#    print("output average count for unique objects in the last second: ", average_output_count)
#    print("---------end of second ---------------")
    

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
#    print("Array for the outputs of each frame ", output_arrays)
#    print("Array for output count for unique objects in each frame : ", count_arrays)
#    print("Output average count for unique objects in the last minute: ", average_output_count)
#    print("------------END OF A MINUTE --------------")




detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execute_path, "yolo.h5"))
detector.loadModel()


#custom_objects = detector.CustomObjects(person=True, stop_sign=True, car=True, traffic_light=True, truck=True, motorcycle=True )

#video_path = detector.detectObjectsFromVideo(custom_objects = custom_objects, input_file_path= os.path.join(execute_path, "Day_Sunny_Town_4.mp4"),
 #                                            output_file_path = os.path.join(execute_path, "Day_Sunny_Town_4_detected_1"),frames_per_second=29, log_progress=True)


video_path = detector.detectObjectsFromVideo(input_file_path= os.path.join(execute_path, "Day_Sunny_Town_4.mp4"),
                                             output_file_path = os.path.join(execute_path, "Day_Sunny_Town_4_detected_1"),frames_per_second=10,
                                             per_second_function=forSeconds,per_frame_function=forFram, per_minute_function=forMinute,minimum_percentage_probability=30,  log_progress=True)


print(video_path)