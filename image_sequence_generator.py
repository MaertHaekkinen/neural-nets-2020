import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from load_dataset_numpy import window
from sklearn.preprocessing import LabelBinarizer
import random
from utils import labels_to_binary


class ImageSequenceGenerator:
    
    def __init__(self):
        self.lb = LabelBinarizer()
        labels = set(['BOO', 'BLO', 'BOR', 'BLR', 'OLR', 'OLO', 'OOR', 'OOO'])
        self.lb.fit(list(labels))
 
    # label_type: "binary" or "categorical"
    def png_image_generator(self, path, bs, folder_list, difficulty="All", sequence_limit=16, resize_dimension = 128, label_type = "categorical", aug=None):    
        f = open("{0}/{1}.txt".format(path, difficulty)) 
    
        while True:
            X_data = []
            Y_data = []

            for x in range(len(folder_list)):
                folder = random.choice(folder_list)

                label = folder.split("_")[-2]

                folder += '/difference'
                images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                
                img_list = [] #np.empty((16, 227,227,3)) # images from all the sequences
                flow_path_list = []
                warped_path_list = []
                diff_path_list = []

                # split the images into sequneces of length 16
                #(e.g. folder contains 20 images, then first seq is 1-16, second seq 2-17, third seq 3-18 etc)
                rnd = random.randint(0,max((len(images)-sequence_limit),0))
                try:
                    each = images[rnd:rnd+sequence_limit]
                except IndexError:
                    each = images[rnd:len(images)-1]
                img_seq_list = [] # only images from one 16 image sequence,  size will be (16, 227,227,3)
                one_images_seq = np.array(each) # 1-16, 2-17, etc

                # read each image to numpy sequence
                for img in one_images_seq:
                    img_load = load_img(img, target_size = (resize_dimension,resize_dimension))
                    img_array = img_to_array(img_load)
                    img_seq_list.append(img_array)


                X_data.append(np.asarray(img_seq_list)) 
                Y_data.append(label)#folder_components[-2])
                if (len(X_data) == bs):                        
                    if label_type == "categorical":
                        Y_data = self.lb.transform(np.array(Y_data))
                    elif label_type == "binary":
                        Y_data = labels_to_binary(Y_data)
                    else:
                        print('Invalid label type!')
                    yield np.asarray(X_data), np.asarray(Y_data)
                    X_data=[]
                    Y_data=[]