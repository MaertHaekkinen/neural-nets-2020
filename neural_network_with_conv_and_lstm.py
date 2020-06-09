 # coding=utf-8
    
import keras
import numpy as np
import cv2
import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import load_img,img_to_array
from keras.layers import Dense, LSTM, Input, Conv2D, Dense, LSTM,MaxPooling2D , Flatten, TimeDistributed,Activation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from create_train_and_validation import create_train_and_validation_set
from load_dataset_numpy import window

import tensorflow as tf
print('is gpu available?', tf.test.is_gpu_available())


#############################################################################
# Setting up parameters
#############################################################################
path = "./rear_signal_dataset"
difficulty = "Easy"
label_type = "binary" # two options: binary or categorical

# parameters for model
im_size = 128
time_steps = 16  # len of image sequence
channels = 3

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 2  # 20
BS = 8  # standard batch size is 32 või 64, but kernal dies with BS larger than 8

print('Info\ndifficulty: {0} \nlabel type: {1} \nimage size: {2} \nepochs: {3} \nbatch size: BS {4}'.format(difficulty,label_type,im_size, NUM_EPOCHS, BS))

#############################################################################
# BINARY MODEL
#############################################################################

#modelB for binary response
modelB = Sequential()
modelB.add(TimeDistributed(Conv2D(filters=96, kernel_size=7,  strides=2, padding='valid'), input_shape=(time_steps,im_size,im_size,channels))) # first input shape is the len of seq
modelB.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
modelB.add(TimeDistributed(Conv2D(filters=384, kernel_size=3,  strides=2, padding='valid')))
modelB.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
modelB.add(TimeDistributed(Conv2D(filters=512, kernel_size=3, padding='same')))
modelB.add(TimeDistributed(Conv2D(filters=512, kernel_size=3, padding='same')))
modelB.add(TimeDistributed(Conv2D(filters=384, kernel_size=3, padding='same')))
modelB.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
modelB.add(TimeDistributed(Flatten())) #The Flatten layer is only needed because LSTM shape should have one dimension per input.
modelB.add(TimeDistributed(Dense(4096)))
modelB.add(LSTM(256, return_sequences=False))
#When return_sequences=True, the output shape is (batch, timeSteps, outputFeatures)
#When return_sequences=False, the output shape is (batch, outputFeatures)

if label_type == 'binary':
    modelB.add((Dense(3))) # for categorical use Dense(8), for binary use Dense(3)
elif label_type == 'categorical':
    modelB.add((Dense(8)))
else:
    print('Invalid label type')
modelB.add((Activation('softmax')))
print(modelB.summary())



#############################################################################
# Convert labels to binary or categorical 
#############################################################################

if label_type == 'binary':
    mlb = MultiLabelBinarizer(classes=['B','L','R'])

    def labels_to_binary(ini_labels):
        """
        Takes as input list of labels (e.g. ['BOO', 'BLO', 'BOR'])
        Outputs numpy ndarray of the labels in binary (e.g. [[1 0 0] [1 1 0] [1 0 1])
        """
        labels = []
        for label in ini_labels:
            label_split = list(label) # ['BLO'] -> ['B','L','O']
            labels.append(label_split) # [['B','L','O']]

        return mlb.fit_transform(labels)

elif label_type == 'categorical':
    #Encoding the labels
    labels = set(['BOO', 'BLO', 'BOR', 'BLR', 'OLR', 'OLO', 'OOR', 'OOO'])
    lb = LabelBinarizer()
    lb.fit(list(labels))
else:
    print('Invalid label type')


#############################################################################
# Train, validation and test creation
#############################################################################

train_folder_list, valid_folder_list, test_folder_list, count_train_seq, count_valid_seq, count_test_seq = create_train_and_validation_set(path, difficulty)

#############################################################################
## Fit generator
#############################################################################


class ImageSequenceGenerator:
    
    def __init__(self):
        self.folder_count = 0
        self.seq_count = 0
        self.image_count = 0
        self.folder_list = []


    # label_type: "binary" or "categorical"
    def png_image_generator(self, path, bs, mode="train",difficulty="All", sequence_limit=16, resize_dimension = 128, label_type = "categorical", aug=None):    
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
                folder += "/difference"
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
    

#############################################################################
## Training the model
#############################################################################

# initialize the total number of training and testing sequences
# ss = small sample
ss = 0.01
NUM_TRAIN_SEQ = round(count_train_seq*ss,0)
NUM_VALID_SEQ = round(count_valid_seq*ss,0)
NUM_TEST_SEQ = round(count_test_seq*ss,0)

img_seq_gen_train = ImageSequenceGenerator()
img_seq_gen_valid = ImageSequenceGenerator()

trainGenB = img_seq_gen_train.png_image_generator(path, bs=BS, mode="train", difficulty=difficulty, label_type = label_type, aug=None)
validGenB = img_seq_gen_valid.png_image_generator(path, bs=BS, mode="valid", difficulty=difficulty, label_type = label_type, aug=None)


opt = Adam(lr=1e-2)

if label_type == 'binary':
    modelB.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\nBinary model: steps per epoch: {0}, validation steps: {1}'.format(NUM_TRAIN_SEQ // BS, NUM_VALID_SEQ // BS))     
elif label_type == 'categorical':
    modelB.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
    print('\nCategorical model: steps per epoch: {0}, validation steps: {1}'.format(NUM_TRAIN_SEQ // BS, NUM_VALID_SEQ // BS))    
else:
    print('Invalid label type')
    
 
# train the network
print("[INFO] training w/ generator...")
historyB = modelB.fit(
    x=trainGenB,
    steps_per_epoch=NUM_TRAIN_SEQ // BS,
    validation_data=validGenB,
    validation_steps=NUM_VALID_SEQ // BS,
    epochs=NUM_EPOCHS)

#############################################################################
## Saving the model
#############################################################################

model_file_name = 'model_' + difficulty + '_' + label_type + '_epochs' + str(NUM_EPOCHS) + '_bs' + str(BS) + '_imsize' + str(im_size)
print('\nSaving model under name: ', model_file_name)
modelB.save('saved_models/'+model_file_name)

print('Completed with the training')

#############################################################################
## Saving  accuracy and loss plots
#############################################################################

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(historyB.history['accuracy'])
plt.plot(historyB.history['val_accuracy'])
#plt.title('model binary accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('images/accuracy_' + model_file_name + '.png',dpi=200)
plt.show()

# summarize history for loss
plt.plot(historyB.history['loss'])
plt.plot(historyB.history['val_loss'])
#plt.title('model binary loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('images/loss_' + model_file_name + '.png',dpi=200)
plt.show()