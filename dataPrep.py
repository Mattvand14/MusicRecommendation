import json
import numpy as np

#open files with read
with open('features.json', 'r') as f:
    features_list = json.load(f)

with open('labels.json', 'r') as f:
    labels_list = json.load(f)


#extract specific features using numpy ndarray
#ex output array([
#    [0.8, 0.6, 120.0, 0.5],
#    [0.7, 0.8, 130.0, 0.6],
#    [0.6, 0.9, 140.0, 0.7]
#])
#each row represents an individual song, columns represent the features
X = np.array([[f['danceability'], f['energy'], f['tempo']], f['valence'] for f in features_list])


#encode labels as integers
#create the mapping from label to integer and #encode the labels using the created mapping
label_to_int = {label: idx for idx, label in enumerate(labels_list)}
y = [label_to_int[label] for label in labels_list]


