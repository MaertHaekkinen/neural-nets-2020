import random
import os

def get_folder_names_dict(path, difficulty):
    """
    Returns: 
    dict: dictionary, where keys are folder names and 
    values represent the number of image sequences given folder contains
    int: total number of images sequences in all non test folders
    """
    all_not_test_folder_names = {}

    f = open("{0}/{1}.txt".format(path, difficulty))
    content = f.readlines()
    # save all the train folder names and how much sequences the folder holds into all_not_test_folder_names
    for  folder in content:
        # Load the data
        folder_components = folder.split("_")
        folder_components[-1] = folder_components[-1][:-1]
        folder_name = str("_".join(folder_components))
        if("test-" in folder_name):
            continue

        # count the number of images in given folder
        base = str("_".join(folder_components[:-2]))
        folder = "{0}/{1}/{2}".format(path, base, (base + "_" + str(folder_components[-2])))
        folder += "/" + folder_name + "/light_mask"
        images = [folder + "/" + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        # number of sequeces we get from a given folder (e.g. 18 pictures is 3 sequences, 1-16, 2-17, 3-18)
        all_not_test_folder_names[folder_name] = len(images) - 15

    count_img_seq = sum(all_not_test_folder_names.values())
    print('total number of non-test folders is  ', len(all_not_test_folder_names) , ' for difficulty ', difficulty)
    print('total number of image sequences is ', count_img_seq)
    return all_not_test_folder_names, count_img_seq


def create_train_and_validation_set(path, difficulty):
    """
    Returns:
    list: contains folder names belonging to train set
    list: contains folder names belonging to validation set
    int: number of images sequences in train set
    int: number of images sequences in validation set
    """
    # get non test folder names
    all_not_test_folder_names, count_img_seq = get_folder_names_dict(path, difficulty)
    
    
    # each folder contains different amount of sequences
    # randomly start adding FOLDERS to validation set, until we have reach 30%
    folder_names_list = list(all_not_test_folder_names.keys())
    random.shuffle(folder_names_list)

    valid_folder_list = []  # validation folder names
    count_valid_seq = 0 # how many images sequences our validation list currently holds
    
    # creates validation set
    for folder_name in folder_names_list:
        valid_folder_list.append(folder_name)
        folder_names_list.remove(folder_name)

        count_valid_seq += all_not_test_folder_names[folder_name] # find how many sequences folder contains
        # we have gotten enough folder for validation set
        if count_valid_seq > int(count_img_seq*0.3):
            break

    # creates train set
    train_folder_list = folder_names_list # folder names remaining in folder_names_list are train folders
    count_train_seq = 0 # number of sequences in training data
    for train_folder in train_folder_list:
        count_train_seq += all_not_test_folder_names[train_folder]

    print('\nfinal train set contains ', count_train_seq, ' image sequences (', round(count_train_seq/count_img_seq*100), '% )')
    print('final validation set contains ', count_valid_seq, ' image sequences (', round(count_valid_seq/count_img_seq*100),'% )')
    
    return train_folder_list, valid_folder_list, count_train_seq, count_valid_seq

