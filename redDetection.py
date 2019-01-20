#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:43:31 2019

@author: soniabaee
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def realtTimeHSV():
    '''detecting HSV of the video'''
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Sunny_High_1" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    
    def nothing(x):
        pass
    
    # Creating a window for later use
    cv2.namedWindow('result')
    
    # Starting with 100's to prevent error while masking
    h,s,v = 100,100,100
    
    # Creating track bar
    cv2.createTrackbar('h', 'result',0,179,nothing)
    cv2.createTrackbar('s', 'result',0,255,nothing)
    cv2.createTrackbar('v', 'result',0,255,nothing)

    while(1):
    
        _, frame = cap.read()
    
        #converting to HSV
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
        # get info from track bar and appy to result
        h = cv2.getTrackbarPos('h','result')
        s = cv2.getTrackbarPos('s','result')
        v = cv2.getTrackbarPos('v','result')
    
        # Normal masking algorithm
        lower_blue = np.array([h,s,v])
        upper_blue = np.array([180,255,255])
    
        mask = cv2.inRange(hsv,lower_blue, upper_blue)
    
        result = cv2.bitwise_and(frame,frame,mask = mask)
    
        cv2.imshow('result',result)
    
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    
    cv2.destroyAllWindows()
    

def detectRed():
    '''detecting red object in video'''
    
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Sunny_High_1" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    
    while(1):
        
        # Take each frame
        _, frame = cap.read()
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
    
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
    
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()



def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
 
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask = mask)
    return output_image



def showLight():
    '''Show light'''
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Rain_High_1" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        blur_frame = cv2.medianBlur(frame, 3)
        # Our operations on the frame come here
        hsv_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
        
        # Get lower red hue
        lower_red_hue = create_hue_mask(hsv_image, [0, 100, 100], [10, 255, 255])
        
        # Get higher red hue
        higher_red_hue = create_hue_mask(hsv_image, [160, 100, 100], [179, 255, 255])
        full_image = cv2.addWeighted(lower_red_hue, 1.0, higher_red_hue, 1.0, 0.0)
        
        
        # Display the resulting frame
        cv2.imshow('frame',full_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    
    cv2.destroyAllWindows()
  
    
def show_hsv_equalized():
    '''show hsv hist equalized'''
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Rain_Town_2" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    frameNumber = 1
    frameLuminosityInfo = {}
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        blur_frame = cv2.medianBlur(frame, 3)
        # Our operations on the frame come here
        H, S, V = cv2.split(cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        
        frameLuminosityInfo[frameNumber] = {'meanValue': cv2.meanStdDev(V)[0][0][0], 'stdValue': cv2.meanStdDev(V)[1][0][0]}

#        plt.figure(figsize=(10,8))
#        plt.subplot(311)                             #plot in the first cel
#        plt.title("Luminosity Value")
#        plt.hist(np.ndarray.flatten(V), bins=128)
#        plt.show()

        print(frameLuminosityInfo)
        # Display the resulting frame
        cv2.imshow('frame',eq_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frameNumber = frameNumber + 1
    

    # When everything done, release the capture
    cap.release()
    
    cv2.destroyAllWindows()
    

def convertHSV():
    '''convert video to hsv display'''
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Rain_High_1" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        blur_frame = cv2.medianBlur(frame, 3)
        # Our operations on the frame come here
        hsv_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
        
        # Display the resulting frame
        cv2.imshow('frame',hsv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    
    cv2.destroyAllWindows()
    
    
def hsvThreshold():
    
    '''showing HSV threshold'''
    
    directory = "/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/videos/"
    fileName = "Day_Sunny_High_1" + ".mp4"
    
    cap = cv2.VideoCapture(directory + fileName)
    
    def nothing(x):
        pass

    useCamera=False
    
    
    # Create a window
    cv2.namedWindow('image')
    
    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)
    
    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)
    
    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    
    
    
    while(1):
    
        
        ret, img = cap.read()
        output = img
    
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')
    
        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')
    
        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
    
        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img,img, mask= mask)
    
        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax
    
        # Display output image
        cv2.imshow('image',output)
    
        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources

    cap.release()
    cv2.destroyAllWindows()

def main():
    
#    realtTimeHSV()
#    convertHSV()
#    hsvThreshold()
    show_hsv_equalized()

if __name__ == '__main__':
    main()
    