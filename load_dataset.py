import os
import cv2
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img

def load_dataset(path="./rear_signal_dataset", difficulty="Moderate", sequence_limit=16):
    image_count = 0
    resize_dimension = 128
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_diff_paths = []
    X_test_diff_paths = []
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
                os.makedirs(folder+"/difference")
            except:
                pass
            folder += "/light_mask"
            images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            image_count += len(images[0:sequence_limit])
            img_list = []
            diff_path_list = []
            for img in images[0:sequence_limit]:
                img_list.append(cv2.cvtColor(cv2.resize(cv2.imread(img), dsize=(resize_dimension, resize_dimension)), cv2.COLOR_BGR2RGB))
                diff_path_list.append(img.replace('/light_mask','/difference'))
            if("test-" in folder):
                X_test.append(img_list)
                Y_test.append(folder_components[-2])
                X_test_diff_paths.append(diff_path_list)
            else:
                X_train.append(img_list)
                Y_train.append(folder_components[-2])
                X_diff_paths.append(diff_path_list)
    return X_train, Y_train, X_diff_paths, X_test, Y_test, X_test_diff_paths, image_count