
import tensorflow as tf
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

    _,_,h,w,c = X.shape
    X = X.reshape(-1,h,w,c)

    y = np.array(date_labels)
    y = y.reshape(-1)
    #converting to 47 bins:
    y -= 140
    y = np.floor_divide(y,5)

    # with obsticles
    X_filt1 = X[y!=46]
    y_filt1 = y[y!=46]

    # without obstacles
    X_filt2 = X[y==46]
    y_filt2 = y[y==46]

    seedNum = 481

    np.random.seed(seed=seedNum)
    ind = np.random.choice(X_filt2.shape[1], 200)

    X_filt_blnc = np.concatenate([X_filt1, X[ind]]) 
    y_filt_blnc = np.concatenate([y_filt1, y[ind]])

    print(X_filt_blnc.shape)
    print(y_filt_blnc.shape)

    return X_filt_blnc, y_filt_blnc



def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print ("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print ("Writing {} done!".format(result_tf_file))

        

x,y = preprocess_filtering_data(date = '2011_09_26', out_name='train02' )
print('np array exists')
#some manipulations to make tf records func work:
x=x.reshape(-1)
x=np.array([x])
y=np.array([y])

np_to_tfrecords(x, y, './', verbose=True)
print('tf record exists')

