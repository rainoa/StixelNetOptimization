import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from sklearn.utils import shuffle
import scipy.misc
import random

def preprocess_data(dates=None, data_path='datasets/kitti/', dir_path='NexarStixelnet/', 
                    output_dir = 'datasets/stixels/', seedNum = 481 ):
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
    
    labels_train = []
    labels_val = []
    labels_test = [] 
        
    
    homedir = os.path.dirname(os.getcwd())
    data_path = os.path.join(homedir, data_path)
    if dates is None:
        dates = [x[1] for x in os.walk(data_path)]
        dates = dates[0]      
    

    for date in dates:
        print(date)
        serieses = [x[1] for x in os.walk(os.path.join(data_path, date))]
        serieses = serieses[0]                            
        
        for series in serieses:
            print(series)
            rootdir = os.path.join(data_path, date, series, "image_02", 'data')
            
            frames = [x[2] for x in os.walk(rootdir)]
            frames = frames[0]
                       
            frames_shuff = shuffle(frames)
            numFrames = len(frames)
            
            
            # train
            for frame in frames_shuff[:int(numFrames*0.7)]:
                labels_train = save_stixels_return_labels_from_frame(labels_train, date, series, frame, data_path,
                                                                     output_dir = os.path.join(homedir, output_dir, 'train'))
            
            labels_train_df = pd.DataFrame(labels_train, columns=['Name', 'Label', 'Use_stixel'])
            labels_train_df = add_no_obstacles_stixels(labels_train_df, percert = 10)          
            labels_train_df.to_csv(os.path.join(homedir, output_dir, 'train', 'labels_train.csv'), encoding='utf-8', index=False)
            
            
            # val
            for frame in frames_shuff[int(numFrames*0.7):int(numFrames*0.9)]:
                labels_val = save_stixels_return_labels_from_frame(labels_val, date, series, frame, data_path,
                                                                   output_dir = os.path.join(homedir, output_dir, 'val'))
            
            labels_val_df = pd.DataFrame(labels_val, columns=['Name', 'Label', 'Use_stixel'])
            labels_val_df = add_no_obstacles_stixels(labels_val_df, percert = 10)
            labels_val_df.to_csv(os.path.join(homedir, output_dir, 'val', 'labels_val.csv'), encoding='utf-8', index=False)
            
            
            # test
            for frame in frames_shuff[int(numFrames*0.9):]:
                labels_test = save_stixels_return_labels_from_frame(labels_test, date, series, frame, data_path,
                                                                    output_dir = os.path.join(homedir, output_dir, 'test'))
            labels_test_df = pd.DataFrame(labels_test, columns=['Name', 'Label', 'Use_stixel'])
            labels_test_df['Use_stixel'] = 1
            labels_test_df.to_csv(os.path.join(homedir, output_dir, 'test', 'labels_test.csv'), encoding='utf-8', index=False)
   
    return 
      

    
    
def save_stixels_return_labels_from_frame(labels, date, series, frame, data_path, output_dir):
           
    stx_w = 24
    stride = 5
    
    GT_file_path = os.path.join(data_path, 'stixelsGroundTruth' + "." + 'txt')
    ground_truth = pd.read_table(GT_file_path, names=["series_date","series_id","frame_id","x","y","Train_Test"]) #making GT pd.df
    
    rootdir = os.path.join(data_path, date, series, "image_02", 'data')
    
    stx_list = [] #list of np.arrays representing stixels only of this frame 

    impath = os.path.join(rootdir, frame)
    img = mpimg.imread(impath) #img is np.array of the frame
    h, w, c = img.shape
    num_stixels = int(((w - stx_w)/stride) +1)
    
    frame_ground_truth = ground_truth[ground_truth.series_date == date[5:]]
    frame_ground_truth = frame_ground_truth[frame_ground_truth.series_id == int(series[18:21])]
    frame_ground_truth = frame_ground_truth[frame_ground_truth.frame_id == int(frame[:-4])]
                
    for stixel in range(num_stixels):
        
        i = 12+(stixel*5)
        s = img[6:,i-12:i+12,:] #that's the stixel, start from 6 because stixel hieght is 370 and not 376
        imName = series[:-5] + '_frame_' + frame[:-4] +'_stixel_' + str(stixel).zfill(3)
        # save the stixel image
        scipy.misc.imsave(os.path.join(output_dir, imName + '.png'), s)
        
        # calculate the label value
        stx_ground_truth = frame_ground_truth[frame_ground_truth.x > i-12]
        stx_ground_truth = stx_ground_truth[stx_ground_truth.x < i+12]
        if stx_ground_truth.empty == False:
            stx_y = int(stx_ground_truth['y'].median()) #calc the label.
            stx_y -= 140
            stx_y = np.floor_divide(stx_y,5)
            for_use = 1
        else:
            stx_y = 46 #if no obstical in stixel
            for_use = 0
        
        labels.append([imName, stx_y, for_use])
    return labels
    

    
    
def add_no_obstacles_stixels(labels_df, percert = 10):
    num_stx_with_obst = len(labels_df.index[labels_df['Label'] != 46].tolist())    
    no_obst_train_idx = labels_df.index[labels_df['Label'] == 46].tolist()
    use_idx = random.sample(no_obst_train_idx, int(num_stx_with_obst*percert/100))
    for idx in use_idx:
        labels_df.at[idx, 'Use_stixel'] = 1
    return labels_df
    
    