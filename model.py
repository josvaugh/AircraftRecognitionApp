# Amos Endersen, Joseph Vaughn
# Tutorial found on https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
import sys

# change these for your own needs and locations
model_dir =  ''
# images_dir = '/Users/Amos/Desktop/photo-base/'
# this is one that I set up for single photo testing
images_dir = '/Users/Amos/Dropbox/data-science/aircraft-model-test/'
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]

def create_graph():
    with gfile.FastGFile(os.path.join(
        #WAS classify_image_graph_def.pb
        model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []
    create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            print('Processing %s...' % (image))
            try:
                if not gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)
                image_data = gfile.FastGFile(image, 'rb').read()
                predictions = sess.run(next_to_last_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                labels.append(re.split('_\d+', image.split('/')[5].split('-')[0])[0])
                features[ind, :] = np.squeeze(predictions)
                # this next line is to parse out all the names of the photos. It is set up to
                # account for a naming convention of
                # /Users/Amos/directory/directory/directory/directory/Cessna_172-#.jpg

            except:
                print("skipping photo ")
                os.remove(image)

        return features, labels
'''
# the following three lines of code only need to be run once successfully.
# after that, the model is built, and you can comment them out to use it.
# the pickle.load methods load in the saved model.
features,labels = extract_features(list_images)
pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
'''

features = pickle.load(open('features', 'rb'))
labels = pickle.load(open('labels', 'rb'))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

# un-comment these lines and change the naming parse to test a single photo
# in the test_dir directory
photos = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
X_test, y_test = extract_features(photos)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# use if reading in a single photo
print ("IMAGE CLASSIFIED AS A", y_pred[0])

#print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))
