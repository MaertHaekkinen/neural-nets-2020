import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from load_dataset_numpy import window


class ImageGenerator:
    
    def __init__(self):
        self.folder_count = 0
        self.seq_count = 0
        self.image_count = 0
        self.folder_list = []


    #Source:https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    # aug is not used yet
    # label_type: "binary" or "categorical"
    def png_image_generator(self, path, bs, mode="train",difficulty="All", resize_dimension = 128, label_type = "categorical", aug=None):    
        f = open("{0}/{1}.txt".format(path, difficulty))

        if mode == "train":
            self.folder_list = train_folder_list
        elif mode == "valid":
            self.folder_list = valid_folder_list
        elif mode == "test":
            self.folder_list = test_folder_list            
        
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
                #folder += "/light_mask"
                folder += '/warped'
                images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                self.image_count += len(images)
                
                img_list = [] #np.empty((16, 227,227,3)) # images from all the sequences
                flow_path_list = []
                warped_path_list = []
                diff_path_list = []
                #print('\n processing folder {0}, number of images in that folder {1} '.format(self.folder_count,len(images)))

                # split the images into sequneces of length 16
                #(e.g. folder contains 20 images, then first seq is 1-16, second seq 2-17, third seq 3-18 etc)
                for each in window(images, 16):
                    img_seq_list = [] # only images from one 16 image sequence,  size will be (16, 227,227,3)
                    one_images_seq = np.array(each) # 1-16, 2-17, etc

                    # read each image to numpy sequence
                    for img in one_images_seq:
                        img_load = load_img(img, target_size = (resize_dimension,resize_dimension))
                        img_array = img_to_array(img_load)
                        img_seq_list.append(img_array)

                        #flow_path_list.append(img.replace('/light_mask','/flow_fields'))
                        #warped_path_list.append(img.replace('/light_mask','warped'))
                        #diff_path_list.append(img.replace('light_mask','difference'))

                    self.seq_count += 1
                    X_data.append(np.asarray(img_seq_list))
                    Y_data.append(label)#folder_components[-2])
                    #X_data_flow_paths.append(flow_path_list)
                    #X_data_warped_paths.append(warped_path_list)
                    #X_data_diff_paths.append(diff_path_list)
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