import os
import cv2
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img

def load_dataset(path="./rear_signal_dataset", difficulty="Easy", sequence_limit=12):
    image_count = 0
    resize_dimension = 128
    X_train = []
    Y_train = []
    with open("{0}/{1}.txt".format(path, difficulty)) as f:
        content = f.readlines()
        for folder in content:
            # Load the data
            folder_components = folder.split("_")
            folder_components[-1] = folder_components[-1][:-1]
            base = str("_".join(folder_components[:-2]))
            folder = "{0}/{1}/{2}".format(path, base, (base + "_" + str(folder_components[-2])))
            folder += "/" + (str("_".join(folder_components)))
            folder += "/light_mask"
            images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            image_count += len(images[0:sequence_limit])
            img_list = []
            for img in images[0:sequence_limit]:
                img_list.append(cv2.cvtColor(cv2.resize(cv2.imread(img), dsize=(resize_dimension, resize_dimension)), cv2.COLOR_BGR2RGB))
            X_train.append(img_list)
            Y_train.append(folder_components[-2])
    return X_train, Y_train, image_count