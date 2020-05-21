import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array

from itertools import tee

def window(iterable, size):
    """
    Slides over a list with a window given the size. 
    Example:
        Suppose iterable = [1,2,3,4,5] and size = 3.
        Output on first iteration is [1,2,3]
        Output on the second iteration [2,3,4]
        Output on the third iteration [3,4,5]     
    """
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

def load_dataset_numpy(path="./rear_signal_dataset", difficulty="All", sequence_limit=16, resize_dimension = 128):
    """
    Output is numpy array with shape (nr_of_sequences, sequence_limit, resize_dimension, resize_dimenstion, 3)
    """
    image_count = 0
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_train_flow_paths = []
    X_test_flow_paths = []
    X_train_warped_paths = []
    X_test_warped_paths = []
    X_train_diff_paths = []
    X_test_diff_paths = []
    
    folder_count = 0
    with open("{0}/{1}.txt".format(path, difficulty)) as f:
        content = f.readlines()
        for folder in content:
            # Load the data
            folder_components = folder.split("_")
            folder_components[-1] = folder_components[-1][:-1]
            base = str("_".join(folder_components[:-2]))
            folder = "{0}/{1}/{2}".format(path, base, (base + "_" + str(folder_components[-2])))
            folder += "/" + (str("_".join(folder_components)))
            try:
                os.makedirs(folder+"/flow_fields")
            except: 
                pass
            try:
                os.makedirs(folder+"/warped")
            except:
                pass
            try:
                os.makedirs(folder+"/difference")
            except:
                pass
            folder += "/light_mask"
            images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            image_count += len(images)
            img_list = [] #np.empty((16, 128,128,3)) # images from all the sequences
            
            flow_path_list = []
            warped_path_list = []
            diff_path_list = []
            im_seq_count = 0
            print('processing folder {0}, number of images in that folder {1} '.format(folder_count,len(images)))
            
            # split the images into sequneces of length 16
            #(e.g. folder contains 20 images, then first seq is 1-16, second seq 2-17, third seq 3-18 etc)
            for each in window(images, 16):
                img_seq_list = [] # only images from one 16 image sequence,  size will be (16, 128,128,3)
                one_images_seq = np.array(each) # 1-16, 2-17, etc
                
                # read each image to numpy sequence
                for img in one_images_seq:
                    img_load = load_img(img, target_size = (resize_dimension,resize_dimension))
                    img_array = img_to_array(img_load)
                    img_seq_list.append(img_array)
                    
                    flow_path_list.append(img.replace('/light_mask','/flow_fields'))
                    warped_path_list.append(img.replace('/light_mask','warped'))
                    diff_path_list.append(img.replace('light_mask','difference'))
                    
                if("test-" in folder):
                    X_test.append(np.asarray(img_seq_list)) 
                    Y_test.append(folder_components[-2])
                    X_test_flow_paths.append(flow_path_list)
                    X_test_warped_paths.append(warped_path_list)
                    X_test_diff_paths.append(diff_path_list)
                else:
                    X_train.append(np.asarray(img_seq_list)) 
                    Y_train.append(folder_components[-2])
                    X_train_flow_paths.append(flow_path_list)
                    X_train_warped_paths.append(warped_path_list)
                    X_train_diff_paths.append(diff_path_list)
            folder_count +=1
            print('total cumulative number of image sequences: ', np.asarray(X_train).shape[0])
            if folder_count == 4:
                break
    return np.asarray(X_train), np.asarray(Y_train), X_train_flow_paths, X_train_warped_paths, X_train_diff_paths, np.asarray(X_test), np.asarray(Y_test), X_test_flow_paths, X_test_warped_paths, X_test_diff_paths, image_count
