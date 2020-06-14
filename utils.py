from sklearn.preprocessing import MultiLabelBinarizer

def labels_to_binary(ini_labels=['BOO', 'BLO', 'BOR', 'BLR', 'OLR', 'OLO', 'OOR', 'OOO']):
    """
    Takes as input list of labels (e.g. ['BOO', 'BLO', 'BOR'])
    Outputs numpy ndarray of the labels in binary (e.g. [[1 0 0] [1 1 0] [1 0 1])
    """
    mlb = MultiLabelBinarizer(classes=['B','L','R'])
    labels = []
    for label in ini_labels:
        label_split = list(label) # ['BLO'] -> ['B','L','O']
        labels.append(label_split) # [['B','L','O']]
    return mlb.fit_transform(labels)