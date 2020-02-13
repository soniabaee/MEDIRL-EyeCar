import numpy as np
import os
import re
import time
from tqdm import tqdm
import pandas as pd

os.chdir('E:\\Xiang Guo\\Lian\\analysis\\2018_new')
csv_dir = os.path.join('csv_classification')
RawTxt_dir = 'E:\\Xiang Guo\\Lian\\analysis\\2018_new\\RawCsv_EYE\\'


#time syncronization table (manually defined), indicating the start and end driving frame in the video
time_table = pd.read_csv('2.time syncronization.csv')
outpath = 'E:\\Xiang Guo\\Lian\\analysis\\2018_output\\2_gaze_position\\'

frame_img_dir = 'G:\\Resource for Xiang\\Lian Cui experiment\\eyetracking data\\2.New Experiment\\4.analysis\\output\\pics\\'


list_csv_file = os.listdir(csv_dir)
list_txt_file = os.listdir(RawTxt_dir)

# Only the data with raw csv gaze data and the object detection data will be used
files_intersection = list(set(list_csv_file).intersection(list_txt_file))


# rename the filename
files = list(time_table['filename']) 
for j in range(len(files)):
    files[j] = files[j].split('-')[0]
    files[j] = files[j].title()
    files[j] = files[j].replace(' ', '')
    files[j] = files[j].replace('Lane', ' Lane ')
    files[j] = files[j].replace('Base', ' Base ')



'''
def add_gaze_point(data):
    for framenum in data['frameNum']:

        i = round(framenum*txtlength/framelen)
        data['Tracking Ratio [%]'] = np.zeros(len(data))
        
        
        data['Tracking Ratio [%]'][framenum] = txt_file['Tracking Ratio [%]'][i]
        data['Category Binocular'][framenum] = txt_file['Category Binocular'][i]
        data['Index Binocular'][framenum] = txt_file['Index Binocular'][i]
        data['Pupil Diameter Right [mm]'][framenum] = txt_file['Pupil Diameter Right [mm]'][i]
        
       
        
        if (txt_file['Point of Regard Binocular X [px]'][i] == '-' or txt_file['Point of Regard Binocular Y [px]'][i] == '-'):
            print('outlier:',framenum,i)
            continue

        data['gazeX'][framenum] = round(float(txt_file['Point of Regard Binocular X [px]'][i]))
        data['gazeY'][framenum] = round(float(txt_file['Point of Regard Binocular Y [px]'][i]))
'''


# this function is used to conbime the gaze data with the object detection results
def add_gaze_point2(data,txtlength,framelen):
    data.index = range(len(data))
    indexes = round(data['frameNum']*txtlength/framelen)
    indexes.index = range(len(indexes))
    
    data['gazeX'] = list(txt_file['Point of Regard Binocular X [px]'][indexes])
    data['gazeY'] = list(txt_file['Point of Regard Binocular Y [px]'][indexes])
    data['gazeVectorX'] = list(txt_file['Gaze Vector Right X'][indexes])
    data['gazeVectorY'] = list(txt_file['Gaze Vector Right Y'][indexes])
    data['gazeVectorZ'] = list(txt_file['Gaze Vector Right Z'][indexes])
    data['Tracking Ratio [%]'] = list(txt_file['Tracking Ratio [%]'][indexes])
    data['Category Binocular'] = list(txt_file['Category Binocular'][indexes])
    data['Index Binocular'] = list(txt_file['Index Binocular'][indexes])
    data['Pupil Diameter Right [mm]'] = list(txt_file['Pupil Diameter Right [mm]'][indexes])

    
    


for k in tqdm(files_intersection):
    start = time.time()
    # csv_file is the results of detection
    csv_file = pd.read_csv(os.path.join(csv_dir, k))
    trial_txt = k
    if os.path.exists(os.path.join(RawTxt_dir, trial_txt)):
        # txt_file is the raw gaze data
        txt_file = pd.read_csv(os.path.join(RawTxt_dir, trial_txt))
        
        #go through the frame image ditection
        frame_img_subdir = frame_img_dir + k.split('.')[0]
        framelist = os.listdir(frame_img_subdir)

        # number of the images 
        framelen = len(framelist)
        # length of the raw gaze point
        txtlength = len(txt_file)
    
        #find file's time synchronization information
        time_index = files.index(k.split('-')[0].title())
        
        # get the start and end driving frame for the trail in the time syncronization table
        start_frame = time_table['start frame'][time_index]
        end_frame = time_table['experiment end frame'][time_index]
        
        
        
        n =len(csv_file)
    
        #num of the frame
        csv_file['frameNum'] = [int(re.findall("\d+",x)[0]) for x in csv_file['framename']]
        
        # For the tv classification
        tv_track = csv_file[csv_file['class']==1]
        tv_track = tv_track.sort_values(['frameNum'])
        tv_track.index = tv_track['frameNum']
        tv_range = [a and b for a,b in zip(tv_track['frameNum']>=start_frame,tv_track['frameNum']<=end_frame)]
        
        tv_track = tv_track[tv_range]
    
        # vehicle classification
        vehicle_track = csv_file[csv_file['class']==2]
        vehicle_track = vehicle_track.sort_values(['frameNum'])
        vehicle_track.index = vehicle_track['frameNum']
        vehicle_range = [a and b for a,b in zip(vehicle_track['frameNum']>=start_frame,vehicle_track['frameNum']<=end_frame)]
        
        vehicle_track = vehicle_track[vehicle_range]
        
        
        #side mirror classification
        smirror_track = csv_file[csv_file['class']==3]
        smirror_track = smirror_track.sort_values(['frameNum'])
        smirror_track.index = smirror_track['frameNum']
        smirror_range = [a and b for a,b in zip(smirror_track['frameNum']>=start_frame,smirror_track['frameNum']<=end_frame)]
        
        smirror_track = smirror_track[smirror_range]
        
        
        
        if not vehicle_track.empty:
            add_gaze_point2(vehicle_track,txtlength,framelen)
            vehicle_track.columns = ['framename', 'ymin', 'xmin', 'ymax', 'xmax', 'scores', 'class','frameNum','gazeX','gazeY',\
                            'VectorX','VectorY','VectorZ','Tracking_Ratio','Category_Binocular','Index Binocular','Pupil_Diameter']
            vehicle_track.to_csv(outpath + '\\vehicle\\' + k, index = False)
        else:
            print('{} file vehicle doesn\'t exist'.format(trial_txt))    
            
            
        
        if not tv_track.empty:
            # match the classification and the gaze data
            add_gaze_point2(tv_track,txtlength,framelen)
            tv_track.columns = ['framename', 'ymin', 'xmin', 'ymax', 'xmax', 'scores', 'class','frameNum','gazeX','gazeY',\
                            'VectorX','VectorY','VectorZ','Tracking_Ratio','Category_Binocular','Index Binocular','Pupil_Diameter']
            tv_track.to_csv(outpath + '\\tv\\' + k, index = False)
        else:
            print('{} file tv doesn\'t exist'.format(trial_txt))
            
            
        
        if not smirror_track.empty:
            add_gaze_point2(smirror_track,txtlength,framelen)
            smirror_track.columns = ['framename', 'ymin', 'xmin', 'ymax', 'xmax', 'scores', 'class','frameNum','gazeX','gazeY',\
                            'VectorX','VectorY','VectorZ','Tracking_Ratio','Category_Binocular','Index Binocular','Pupil_Diameter']
            smirror_track.to_csv(outpath + '\\smirror\\' + k, index = False)
        else:
            print('{} file smirror doesn\'t exist'.format(trial_txt))
            
            
    else:
        print('{} file doesn\'t exist'.format(trial_txt))
        continue
    
    
    end =  time.time()
    print("Execution Time for {} is :".format(k), end - start, ' Seconds')



'''
#determine if any file not included in time_table

for k in range(len(list_csv_file)):
    if (len(time_table['filename'][time_table['filename'] ==  list_csv_file[k].split('.')[0]]) == 0):
        print(list_csv_file[k])
    else:
        print('ok')





def add_gaze_point(data):
    for framenum in data['frameNum']:

        i = round(framenum*txtlength/framelen)
        if (txt_file['Point of Regard Binocular X [px]'][i] == '-' or txt_file['Point of Regard Binocular Y [px]'][i] == '-'):
            print('outlier:',framenum,i)
            continue

        data['gazeX'][framenum] = round(float(txt_file['Point of Regard Binocular X [px]'][i]))
        data['gazeY'][framenum] = round(float(txt_file['Point of Regard Binocular Y [px]'][i]))
        
'''





        
