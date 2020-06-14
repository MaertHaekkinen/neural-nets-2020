import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array

from itertools import tee
    
def add_parts(img, new_img):
    img[0:64,0:128]=img[0:64,0:128]
    img[64:128,0:128] = new_img[32:96,0:128]
    return img[0:129,0:128]

def add_combo_images(path="./rear_signal_dataset", difficulty="All", sequence_limit=16, resize_dimension = 128):
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
                os.makedirs(folder+"/combination")
            except:
                pass
            folder += "/warped"
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
            for img in images:
                img_load = load_img(img, target_size = (resize_dimension,resize_dimension))
                img_array = img_to_array(img_load)
                difference_path = img.replace('/warped','/difference')
                difference_path = difference_path.replace('.jpg','.png')
                difference_load = load_img(difference_path, target_size = (resize_dimension,resize_dimension))
                difference_array = img_to_array(difference_load)
                new_img = Image.fromarray(add_parts(img_array, difference_array).astype(np.uint8))
                stre = img.replace('warped','combination')
                print(stre)
                new_img.save(img.replace('warped','combination'))
add_combo_images()

                    