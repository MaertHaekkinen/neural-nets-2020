import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from load_dataset_numpy import window
from sklearn.preprocessing import LabelBinarizer
import random
from utils import labels_to_binary


class ImageSequenceGeneratorOld:
    
    def __init__(self):
        self.folder_count = 0
        self.seq_count = 0
        self.image_count = 0
        self.folder_list = []


    # label_type: "binary" or "categorical"
    def png_image_generator(self, path, bs, folder_list, difficulty="All", sequence_limit=16, resize_dimension = 128, label_type = "categorical", aug=None):    
        f = open("{0}/{1}.txt".format(path, difficulty))
        self.folder_list = folder_list            
        
        while True:
            X_data = []
            Y_data = []
            X_data_flow_paths = []
            X_data_flow_paths = []
            X_data_warped_paths = []
            X_data_warped_paths = []
            X_data_diff_paths = []
            X_data_diff_paths = []

            for folder in self.folder_list:
                label = folder.split("_")[-2]
                folder += '/combination'
                images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                self.image_count += len(images)
                
                img_list = [] #np.empty((15, 227,227,3)) # images from all the sequences
                flow_path_list = []
                warped_path_list = []
                diff_path_list = []

                # split the images into sequneces of length 15
                #(e.g. folder contains 20 images, then first seq is 1-15, second seq 2-17, third seq 3-18 etc)
                for each in window(images, 16):
                    img_seq_list = [] # only images from one 15 image sequence,  size will be (15, 227,227,3)
                    one_images_seq = np.array(each) # 1-15, 2-17, etc

                    # read each image to numpy sequence
                    for img in one_images_seq:
                        img_load = load_img(img, target_size = (resize_dimension,resize_dimension))
                        img_array = img_to_array(img_load)
                        img_seq_list.append(img_array)

                    self.seq_count += 1
                    X_data.append(np.asarray(img_seq_list)) 
                    Y_data.append(label)#folder_components[-2])
                    if (len(X_data) == bs):                        
                        if label_type == "categorical":
                            Y_data = lb.transform(np.array(Y_data))
                        elif label_type == "binary":
                            Y_data = labels_to_binary(Y_data)
                        else:
                            print('Invalid label type!')
                        yield np.asarray(X_data), np.asarray(Y_data)
                        X_data=[]
                        Y_data=[]

                self.folder_count +=1