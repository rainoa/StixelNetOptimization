import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def preprocess_filtering_data(date,out_name, serieses = [1], data_path='/datasets/kitti/' , dir_path='/NexarStixelnet/' ):
    '''date is a string like this: '2011_09_26'.
    TODO: iterate thru different series
    TODO: implement for different dates
    TODO: implement for random "no obstical"
 '''
    serieses = serieses
    stx_w = 24
    stride = 5

    date_labels = [] #future y. will be a list of lists of labels by frames
    date_stx_list = [] #future x. will ce a list of lists of np.arrays by frames
    stx_with_obj = [] #list of stixels # with objext


    homedir = os.path.expanduser('~')
    GT_file_path = os.path.join(homedir, data_path[1:], 'stixelsGroundTruth' + "." + 'txt')
    ground_truth = pd.read_table(GT_file_path, names=["series_date","series_id","frame_id","x","y","Train_Test"]) #making GT pd.df
    homedir = os.path.expanduser('~')
    for series in serieses:
        print(series)
        rootdir =homedir + data_path +  str(date) + '/'+  str(date) + '_drive_000'+ str(series) + '_sync/image_02/data/' #root for dataset

        for subdir, dirs, files in os.walk(rootdir):

            frame_list = files

            for frame in frame_list: #iterate thru frames, in each frame divide to stixles and determine label of stixel by MEDIAN
                stx_list = [] #list of np.arrays representing stixels only of this frame 

                path = rootdir + '/' + frame
                img=mpimg.imread(path) #img is np.array of the frame
                h, w, c = img.shape
                num_stixels = int(((w - stx_w)/stride) +1)

                for stixel in range(num_stixels):
                    i = 12+(stixel*5)
                    s = img[5:,i-12:i+12,:] #thats the stixel
                    stx_list.append(s)
                date_stx_list.append(stx_list)

                #labeling:
                #filtering df according to frame
                frame_ground_truth = ground_truth[ground_truth.series_date == date[5:]]
                frame_ground_truth = frame_ground_truth[frame_ground_truth.series_id == series]
                frame_ground_truth = frame_ground_truth[frame_ground_truth.frame_id == int(frame[:-4])]
                frame_ground_truth = frame_ground_truth[frame_ground_truth.y > 6]

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
                    stx_y_list.append(stx_y)
                    i+=1
                date_labels.append(stx_y_list)

    X = np.array(date_stx_list)
    print(X.shape)

    _,_,h,w,c = X.shape
    X = X.reshape(-1,h,w,c)

    y = np.array(date_labels)
    y = y.reshape(-1)
    #converting to 47 bins:
    y -= 140
    y = np.floor_divide(y,5)

    X_filt1 = X[y!=46]
    y_filt1 = y[y!=46]
    X_filt2 = np.concatenate([X_filt1, X[:100]])
    y_filt2 = np.concatenate([y_filt1, y[:100]])

    print(X_filt2.shape)
    print(y_filt2.shape)

    np.save(homedir+data_path+'X'+out_name, X_filt2)
    np.save(homedir+data_path+'y'+out_name, y_filt2)
    print('saved')
    return True


#preprocess_filtering_data(date = '2011_09_26', out_name='train02' )

