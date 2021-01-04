# import necessary libraries
import cv2
import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
from imageai.Detection import VideoObjectDetection
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pickle
from . import instanceSegmentation

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}
FILENAME = ""

def combineCSV():
    '''
        here we combined all the data in a single csv file
    '''

    path = os.getcwd() + "/FrameObj/"

    files = glob.gob(path + "*.txt")
    Names = [f.split("_output")[0] for f in files]

    for n in Names:
        listFiles = [f for f in os.listdir(path) if f.startswith(n)]
        combined = pd.concat([pd.read_csv(f) for f in listFiles])
        combined.to_csv()


def generateFrame(videoName):
    '''
        here we extract all the frames of each video to assign
        visual attention allocation map on them
    '''
    if len(videoName) == 0:
        os.chdir("./medirl-master/Code/")
        VideoDir = "./medirl-master/videos/crash-video"
        videos = glob.glob(VideoDir + '/*.mp4')
        pathOut = "./medirl-master/videos/crash-video/output"

        for v in videos:
            # v = "./medirl-master/videos/crash-video/2934487_detected.avi"
            vidcap = cv2.VideoCapture(v)
            success,image = vidcap.read()
            instance_segmentation_api(image, 0.75, rect_th=1, text_size=0.3, text_th=1)
            folder = "Frames"
            directory = "/"+ pathOut + "/" + folder
            if not os.path.exists(directory):
                os.makedirs(directory)
            count = 0
            success = True
            while success:
                cv2.imwrite(os.path.join(directory, "frame{:d}.png".format(count)), image)
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
    else:

        v = videoName
        vidcap = cv2.VideoCapture(v)
        success,image = vidcap.read()
        instance_segmentation_api(image, 0.75, rect_th=1, text_size=0.3, text_th=1)
        folder = "Frames"
        directory = "/"+ pathOut + "/" + folder
        if not os.path.exists(directory):
            os.makedirs(directory)
        count = 0
        success = True
        while success:
            cv2.imwrite(os.path.join(directory, "frame{:d}.png".format(count)), image)
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1


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
    '''
        detecting object for each frame
    '''
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
    return(fileName.split(".")[0] + "_detected.avi")



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    

def detectRed():
    '''
    detecting red object in video
    '''
    os.chdir("./medirl-master/Code/")
    VideoDir = "./medirl-master/videos/crash-video"
    videos = glob.glob(VideoDir + '/*.mp4')
    
    for v in videos:
        cap = cv2.VideoCapture(v)
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
    '''
        Show light
    '''
    
    os.chdir("./medirl-master/Code/")
    VideoDir = "./medirl-master/videos/crash-video"
    videos = glob.glob(VideoDir + '/*.mp4')
    
    for v in videos:
    
        cap = cv2.VideoCapture(v)
    
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
            # Convert image to grayscale
            image_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

            
            # Display the resulting frame
            cv2.imshow('frame',image_gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # When everything done, release the capture
        cap.release()
        
        cv2.destroyAllWindows()
  
    
def show_hsv_equalized(directory, fileName):
    '''
        show hsv hist equalized
    '''
    
    cap = cv2.VideoCapture(directory + fileName)
    frameNumber = 1
    frameLuminosityInfo = {}
    
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            frameLuminosityInfo = pd.DataFrame(frameLuminosityInfo)
            frameLuminosityInfo.to_csv(directory + fileName + ".csv")
            return frameLuminosityInfo
            break
        
        blur_frame = cv2.medianBlur(frame, 3)
        # Our operations on the frame come here
        H, S, V = cv2.split(cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        
        frameLuminosityInfo[frameNumber] = {'meanValue': cv2.meanStdDev(V)[0][0][0], 'stdValue': cv2.meanStdDev(V)[1][0][0]}



        # Display the resulting frame
        cv2.imshow('frame',eq_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        frameNumber = frameNumber + 1
    
    # When everything done, release the capture
    cap.release()
    
    cv2.destroyAllWindows()
    
    
    
def hsvThreshold():
    
    '''showing HSV threshold'''
    
    os.chdir("./medirl-master/Code/")
    VideoDir = "./medirl-master/videos/crash-video"
    videos = glob.glob(VideoDir + '/*.mp4')
    
    for v in videos:
        cap = cv2.VideoCapture(v)
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
    
    
def LuminosityStat(directory):
    """ the basic statistical things about luminosity of the videos"""
    with open(directory+'luminosity.pkl', 'rb') as input:
        luminosity = pickle.load(input)
    pickTimeVideo = {}
    for key in luminosity:
        frameLumnosity = luminosity[key]
        frameLumnosity = frameLumnosity.T
        frameSec = frameLumnosity.shape[0]/(30)
        pickTimeVideo[key] = (frameLumnosity['meanValue'].idxmax())/frameSec
        
    return pickTimeVideo;   
    
def VFE():

    os.chdir("./medirl-master/Code/")
    VideoDir = "./medirl-master/videos/crash-video"
    videos = glob.glob(VideoDir + '/*.mp4')
    pathOut = "./medirl-master/videos/crash-video/output"

    for v in videos:
        objectDection(v, VideoDir)
        generateFrame(v, VideoDir)
        combineCSV(v, VideoDir)
        showLight(v, VideoDir)
        show_hsv_equalized(v, VideoDir)
        hsvThreshold(v, VideoDir)
        LuminosityStat(v, VideoDir)
        detectRed(v, VideoDir)
