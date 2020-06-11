# import necessary libraries
import cv2
import numpy as np
from keras.preprocessing import image
from VFE.VFE import *
import process_image as ip
from . import maskModel
import torch.nn as nn
import torch
from . import convolution_bilstm
import os
import argparse
import datetime
from . import config as configurable
import torch
import torchtext.data as data
from . import train_cnn
from . import train_lstm
import multiprocessing as mu
import shutil
import random

# solve encoding
from imp import reload
import sys
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
from DataUtils.Common import seed_num, pad, unk
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)



class InferenceConfig():
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE_SHAPES=np.array([[256, 256], [128, 128],  [ 64,  64],  [ 32,  32],  [ 16,  16]])
    BACKBONE_STRIDES=[4, 8, 16, 32, 64]
    BATCH_SIZE=1
    BBOX_STD_DEV=[ 0.1,  0.1,  0.2,  0.2]
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=0.6 #0.5
    DETECTION_NMS_THRESHOLD=0.3
    IMAGE_MAX_DIM=1024
    IMAGE_MIN_DIM=800
    IMAGE_PADDING=True
    IMAGE_SHAPE=np.array([1024, 1024,    3])
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE =0.002
    MASK_POOL_SIZE=14
    MASK_SHAPE    =[28, 28]
    MAX_GT_INSTANCES=100
    MEAN_PIXEL      =[ 123.7,  116.8,  103.9]
    MINI_MASK_SHAPE =(56, 56)
    NAME            ="coco"
    NUM_CLASSES     =81
    POOL_SIZE       =7
    POST_NMS_ROIS_INFERENCE =1000
    POST_NMS_ROIS_TRAINING  =2000
    ROI_POSITIVE_RATIO=0.33
    RPN_ANCHOR_RATIOS =[0.5, 1, 2]
    RPN_ANCHOR_SCALES =(32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE =2
    RPN_BBOX_STD_DEV  =np.array([ 0.1,  0.1,  0.2 , 0.2])
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    RPN_NMS_THRESHOLD = 0.3
    STEPS_PER_EPOCH            =1000
    TRAIN_ROIS_PER_IMAGE       =128
    USE_MINI_MASK              =True
    USE_RPN_ROIS               =True
    VALIDATION_STPES           =50
    WEIGHT_DECAY               =0.0001

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def findLane(img):
    cropped_img = ip.area_of_interest(img, [ip.crop_points.astype(np.int32)])
    trans_img  = ip.applyTransformation(cropped_img)
    masked_image = ip.applyMasks(trans_img)
    left_fit, right_fit, _ = ip.slidingWindow(masked_image)
    lane_mask = ip.applyBackTrans(img, left_fit, right_fit)
    img_result = cv2.addWeighted(img, 1, lane_mask, 1, 0)
    return img_result

def process_video(neural_net, input_img):

    img = cv2.resize(input_img, (1024, 1024))
    img = image.img_to_array(img)
    results = neural_net.detect([img], verbose=0)
    r = results[0]
    final_img = visualize_car_detection.display_instances2(img, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
    inp_shape = image.img_to_array(input_img).shape
    final_img = cv2.resize(final_img, (inp_shape[1], inp_shape[0]))

    return final_img


import cv2
import numpy as np
from moviepy.editor import VideoFileClip

prev_frames = []
crop_points = np.float32([[0 , 720],
                         [1280 , 720],
                         [750 , 470],
                         [530 , 470]])

trans_points = np.float32([[320 , 720],
                         [960 , 720],
                         [960 , 0],
                         [320 , 0]])

def applyBackTrans(img, left_fit, right_fit):
    ploty = np.linspace(0, 719, num=720)
    # Calculate left and right x positions
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Defining a blank mask to start with
    polygon = np.zeros_like(img)

    # Create an array of points for the polygon
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the polygon in blue
    cv2.fillPoly(polygon, np.int_([pts]), (0, 0, 255))

    # Calculate top and bottom distance between the lanes
    top_dist = right_fitx[0] - left_fitx[0]
    bottom_dist = right_fitx[-1] - left_fitx[-1]

    # Add the polygon to the list of last frames if it makes sense
    if len(prev_frames) > 0:
        if top_dist < 300 or bottom_dist < 300 or top_dist > 500 or bottom_dist > 500:
            polygon = prev_frames[-1]
        else:
            prev_frames.append(polygon)
    else:
        prev_frames.append(polygon)

    # Check that the new detected lane is similar to the one detected in the previous frame
    polygon_gray = cv2.cvtColor(polygon, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_frames[-1], cv2.COLOR_RGB2GRAY)
    non_similarity = cv2.matchShapes(polygon_gray,prev_gray, 1, 0.0)
    if non_similarity > 0.002:
        polygon = prev_frames[-1]

    # Calculate the inverse transformation matrix
    M_inv = cv2.getPerspectiveTransform(trans_points, crop_points)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    image_backtrans = cv2.warpPerspective(polygon, M_inv, (img.shape[1], img.shape[0]))

    # Return the 8-bit mask
    return np.uint8(image_backtrans)


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def slidingWindow(img):
    # Window settings
    window_width = 50
    window_height = 100
    # How much to slide left and right for searching
    margin = 30

    # Store the (left,right) window centroid positions per level
    window_centroids = []
    # Create our window template that we will use for convolutions
    window = np.ones(window_width)

    # Find the starting point for the lines
    l_sum = np.sum(img[int(3*img.shape[0]/5):,:int(img.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(img[int(3*img.shape[0]/5):,int(img.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(img.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        # Find the best left centroid by using past left center as a reference
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    # If we have found any window centers, print error and return
    if len(window_centroids) == 0:
        print("No windows found in this frame!")
        return

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(img)
    r_points = np.zeros_like(img)

    # Go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channle
    template = np.array(cv2.merge((template, template, template)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((img, img, img)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # Extract left and right line pixel positions
    leftx = np.nonzero(l_points)[1]
    lefty = np.nonzero(l_points)[0]
    rightx = np.nonzero(r_points)[1]
    righty = np.nonzero(r_points)[0]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Return left and right lines as well as the image
    return left_fit, right_fit, output

def area_of_interest(img, points):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, points, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def applyTransformation(img):
    M = cv2.getPerspectiveTransform(crop_points, trans_points)
    transformed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return transformed

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, thresh_min=0, thresh_max=255):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output

def applyMasks(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Apply sobel in x direction on L and S channel
    l_channel_sobel_x = abs_sobel_thresh(l_channel,'x', 20, 200)
    s_channel_sobel_x = abs_sobel_thresh(s_channel,'x', 60, 200)
    sobel_combined_x = cv2.bitwise_or(s_channel_sobel_x, l_channel_sobel_x)

    # Apply magnitude sobel
    l_channel_mag = mag_thresh(l_channel, 80, 200)
    s_channel_mag = mag_thresh(s_channel, 80, 200)
    mag_combined = cv2.bitwise_or(l_channel_mag, s_channel_mag)

    # Combine all the sobel filters
    sobel_mask = cv2.bitwise_or(mag_combined, sobel_combined_x)

    # Mask out the desired image and filter image again
    sobel_mask = area_of_interest(sobel_mask, np.array([[(330, 0),(950, 0), (950, 680), (330, 680)]]))

     # Convert to HLS and extract S and V channel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define color thresholds in HSV
    white_low = np.array([[[0, 0, 210]]])
    white_high = np.array([[[255, 30, 255]]])

    yellow_low = np.array([[[18, 80, 80]]])
    yellow_high = np.array([[[30, 255, 255]]])

    # Apply the thresholds to get only white and yellow
    white_mask = cv2.inRange(img_hsv, white_low, white_high)
    yellow_mask = cv2.inRange(img_hsv, yellow_low, yellow_high)

    # Bitwise or the yellow and white mask
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)

    mask_combined = np.zeros_like(sobel_mask)
    mask_combined[(color_mask>=.5)|(sobel_mask>=.5)] = 1
    return mask_combined



class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

%%%%%%%%%%%%%%%%%%5
# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : main.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}




d

def save_arguments():
    shutil.copytree("./Config", "./snapshot/" + config.mulu + "/Config")


def update_arguments():
    config.lr = config.learning_rate
    config.init_weight_decay = config.weight_decay
    config.init_clip_max_norm = config.clip_max_norm
    config.embed_num = len(config.text_field.vocab)
    config.class_num = len(config.label_field.vocab) - 1
    config.paddingId = config.text_field.vocab.stoi[pad]
    config.unkId = config.text_field.vocab.stoi[unk]
    if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
        config.embed_num_mui = len(config.static_text_field.vocab)
        config.paddingId_mui = config.static_text_field.vocab.stoi[pad]
        config.unkId_mui = config.static_text_field.vocab.stoi[unk]
    # config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]
    print(config.kernel_sizes)
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu = mulu
    config.save_dir = os.path.join(""+config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)


def load_model():
    model = None
    if config.snapshot is None:
        config.CNN_BiLSTM:
        print("loading CNN_BiLSTM model......")
        model = CNN_BiLSTM(config)
        shutil.copy("./convolution_bilstm.py", "./snapshot/" + config.mulu)
        print(model)
    else:
        print('\nLoading model from [%s]...' % config.snapshot)
        try:
            model = torch.load(config.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()
    if config.cuda is True:
        model = model.cuda()
    return model


def start_train(model, train_iter, dev_iter, test_iter):
    """
    :function：start train
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    if config.predict is not None:
        label = train_cnn.predict(config.predict, model, config.text_field, config.label_field)
        print('\n[Text]  {}[Label] {}\n'.format(config.predict, label))
    elif config.test:
        try:
            print(test_iter)
            train_cnn.test_eval(test_iter, model, config)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print("\n cpu_count \n", mu.cpu_count())
        torch.set_num_threads(config.num_threads)
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")
        config.CNN_BiLSTM:
        print("CNN_BiLSTM training start......")
        model_count = train_lstm.train(train_iter, dev_iter, test_iter, model, config)
        print("Model_count", model_count)
        resultlist = []
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt")
            for line in file.readlines():
                if line[:10] == "Evaluation":
                    resultlist.append(float(line[34:41]))
            result = sorted(resultlist)
            file.close()
            file = open("./Test_Result.txt", "a")
            file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
            file.write("\n")
            file.close()
            shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")


def IE():

    frameList = "./medirl-master/videos/crash-video/output"
    os.chdir("./medirl-master/Code/")
    VideoDir = "./medirl-master/videos/crash-video"
    videos = glob.glob(VideoDir + '/*.mp4')
    pathOut = "./medirl-master/videos/crash-video/output"
    #---------------------------------------------------
    #lane change and task
    for video in videos:
        detected_video = VFE.objectDection(pathOut + '/Frame/' + videos)
        videoFrames = VFE.generateFrame(detected_video)
        for frame in videoFrames:
            img_arr = cv2.imread(pathOut + "/Frame/" + frame)
            config = InferenceConfig()
            img_result = findLane(img_arr)
            cv2.imwrite(pathOut + "/Frame/" + frame + "_detected.png", img_result)

    #---------------------------------------------------
    #task-mask
    for video in videos:
        detected_video = VFE.objectDection(pathOut + '/Frame/' + videos)
        videoFrames = VFE.generateFrame(detected_video)
        for frame in videoFrames:
            img_arr    = cv2.imread(frame)
            cropped_img = area_of_interest(img_arr, [crop_points.astype(np.int32)])
            trans_img  = applyTransformation(cropped_img)
            masked_image = applyMasks(trans_img)
            left_fit, right_fit, _ = slidingWindow(masked_image)

            lane_mask = applyBackTrans(img_arr, left_fit, right_fit)

            img_result = cv2.addWeighted(img_arr, 1, lane_mask, 1, 0)

            cv2.imwrite(pathOut + "/Frame/" + frame + "_detected.png", img_result)


    #---------------------------------------------------
    #intent
    # clstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=5, step=9, effective_step=[2, 4, 8])
    # lstm_outputs = clstm(cnn_features)
    # hidden_states = lstm_outputs[0]

    # loss_fn = torch.nn.MSELoss()

    # input = Variable(torch.randn(1, 512, 64, 32)).cuda()
    # target = Variable(torch.randn(1, 32, 64, 32)).double().cuda()

    # output = convlstm(input)
    # output = output[0][0].double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./config.cfg")
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)
    if config.cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())


    
    model = load_model()
    start_train(model, train_iter, dev_iter, test_iter)
