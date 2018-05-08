import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from sklearn.utils import shuffle


def preprocess_data(dates=None, data_path='datasets/kitti/', dir_path='NexarStixelnet/', seedNum = 481 ):
    '''
    This function goes through all the KITTI database, and creates stixels with the GT data  (https://sites.google.com/view/danlevi/datasets?authuser=0). 
    It splits the frames as follows:
        70% - train set
        20% - validation set
        10% - test set
    Then from each frame the stixels are extracted.
    In the train set, we take only the stixels with obstacles and add 10% of the number of stixels with stixel that don't have obstacles.
    Validation and test sets contain all the stixels.
    
    dates is None or a list of strings like this: ['2011_09_26'].
    '''
    
    X_train_unprocess = None 
    y_train_unprocess = None
    X_val_unprocess = None 
    y_val_unprocess = None
    X_test_unprocess = None 
    y_test_unprocess = None
    
    
    homedir = os.path.dirname(os.getcwd())
    data_path = os.path.join(homedir, data_path)
    if dates is None:
        dates = [x[1] for x in os.walk(data_path)]
        dates = dates[0]      
    
    for date in dates:
        print(dates)
        serieses = [x[1] for x in os.walk(os.path.join(data_path, date))]
        serieses = serieses[0]                            
        
        for series in serieses:
            rootdir = os.path.join(data_path, date, series, "image_02", 'data')
            
            frames = [x[2] for x in os.walk(rootdir)]
            frames = frames[0]
            
            frames_stx_list = []
            frames_labels = []
            frames_stx_list, frames_labels = get_stixels_from_frames(date,series, frames, data_path)
            X_shuff, y_shuff = shuffle(frames_stx_list, frames_labels, random_state=seedNum)
            
            #calculate in which indices to cut
            numStxInFrames = len(X_shuff)
            if X_train_unprocess is None:
                X_train_unprocess = np.array(X_shuff[:int(numStxInFrames*0.7)]) 
                y_train_unprocess = np.array(y_shuff[:int(numStxInFrames*0.7)])
                X_val_unprocess = np.array(X_shuff[int(numStxInFrames*0.7):int(numStxInFrames*0.9)])  
                y_val_unprocess = np.array(y_shuff[int(numStxInFrames*0.7):int(numStxInFrames*0.9)]) 
                X_test_unprocess = np.array(X_shuff[int(numStxInFrames*0.9):]) 
                y_test_unprocess = np.array(y_shuff[int(numStxInFrames*0.9):]) 
            else:
                X_train_unprocess = np.concatenate((X_train_unprocess, np.array(X_shuff[:int(numStxInFrames*0.7)])))
                y_train_unprocess = np.concatenate((y_train_unprocess, np.array(y_shuff[:int(numStxInFrames*0.7)])))
                X_val_unprocess = np.concatenate((X_val_unprocess, np.array(X_shuff[int(numStxInFrames*0.7):int(numStxInFrames*0.9)])))
                y_val_unprocess = np.concatenate((y_val_unprocess, np.array(y_shuff[int(numStxInFrames*0.7):int(numStxInFrames*0.9)]))) 
                X_test_unprocess = np.concatenate((X_test_unprocess, np.array(X_shuff[int(numStxInFrames*0.9):]))) 
                y_test_unprocess = np.concatenate((y_test_unprocess, np.array(y_shuff[int(numStxInFrames*0.9):])))
    
      
    X_train, y_train = preprocess_stixels(X_train_unprocess, y_train_unprocess, seedNum, is_training = True)
    X_val, y_val = preprocess_stixels(X_val_unprocess, y_val_unprocess, seedNum, is_training = False)
    X_test, y_test = preprocess_stixels(X_test_unprocess, y_test_unprocess, seedNum, is_training = False)
    # To remove stixels with no obstacles, change is_training to True
   
    return X_train, y_train, X_val, y_val, X_test, y_test
      

def get_stixels_from_frames(date, series, frames, data_path):
    stx_w = 24
    stride = 5
    frames_stx_list = []
    frames_labels = []
    GT_file_path = os.path.join(data_path, 'stixelsGroundTruth' + "." + 'txt')
    ground_truth = pd.read_table(GT_file_path, names=["series_date","series_id","frame_id","x","y","Train_Test"]) #making GT pd.df
    
    rootdir = os.path.join(data_path, date, series, "image_02", 'data')
    
    for frame in frames: #iterate thru frames, in each frame divide to stixles and determine label of stixel by MEDIAN
        stx_list = [] #list of np.arrays representing stixels only of this frame 

        impath = os.path.join(rootdir, frame)
        img = mpimg.imread(impath) #img is np.array of the frame
        h, w, c = img.shape
        num_stixels = int(((w - stx_w)/stride) +1)

        for stixel in range(num_stixels):
            i = 12+(stixel*5)
            s = img[6:,i-12:i+12,:] #that's the stixel, start from 6 because stixel hieght is 370 and not 376
            stx_list.append(s)#list of np.arrays representing stixels only of this frame 
        frames_stx_list.append(stx_list)# list of lists of np.arrays representing stixels by frames and dates/series
    
        #labeling:
        #filtering df according to frame
        frame_ground_truth = ground_truth[ground_truth.series_date == date[5:]]
        frame_ground_truth = frame_ground_truth[frame_ground_truth.series_id == int(series[18:21])]
        frame_ground_truth = frame_ground_truth[frame_ground_truth.frame_id == int(frame[:-4])]
                
    
        stx_y_list = []
        i = 0
        for stx in stx_list:
            #filtering df according to stixel:
            stx_ground_truth = frame_ground_truth[frame_ground_truth.x > i-12]
            stx_ground_truth = stx_ground_truth[stx_ground_truth.x < i+12]
            if stx_ground_truth.empty == False:
                stx_y = int(stx_ground_truth['y'].median()) #calc the label. 
            else:
                stx_y = 371 #if no obstical in stixel
            stx_y_list.append(stx_y)# list of ints representing true label of stixels only of this frame 
            i+=1
        frames_labels.append(stx_y_list) # a list of lists representing true label of stixels by frames and dates/series
       
    return frames_stx_list, frames_labels


def preprocess_stixels(X, y, seedNum = 481, is_training = False):
    _,_,h,w,c = X.shape
    X_stx = X.reshape(-1,h,w,c)
    y_stx = y.reshape(-1)
    #converting to 47 bins:
    y_stx -= 140
    y_stx = np.floor_divide(y_stx,5)

    if is_training:
        # with obstacles
        X_obst = X_stx[y_stx!=46]
        y_obst = y_stx[y_stx!=46]
        # without obstacles
        X_no_obst = X_stx[y_stx==46]
        y_no_obst = y_stx[y_stx==46]

    
        numStxWithObst = len(y_obst)
        # seedNum = np.random.randint(1, 1000, size=1)
        np.random.seed(seed=seedNum)
        # taking all stixels with obstacles and another 10% of stixels with no obstacles
        ind = np.random.choice(X_no_obst.shape[1], int(numStxWithObst/10))

        X_stx = np.concatenate([X_obst, X_no_obst[ind]]) 
        y_stx = np.concatenate([y_obst, y_no_obst[ind]])


    return X_stx, y_stx

