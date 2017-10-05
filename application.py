# Amos Endersen
# Tutorial found on http://flask.pocoo.org/docs/0.12/quickstart/
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
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'photos/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

features = pickle.load(open('features', 'rb'))
labels = pickle.load(open('labels', 'rb'))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# change these for your own needs and locations
model_dir =  ''
# images_dir = '/Users/Amos/Desktop/photo-base/'
# this is one that I set up for single photo testing
images_dir = '/Users/Amos/Dropbox/data-science/aircraft-model-test/photos/'
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/<result>', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_file(result=None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            
            photos = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
            X_test, y_test = extract_features(photos)

            clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # use if reading in a single photo
            print ("IMAGE CLASSIFIED AS A", y_pred[0])
            
            os.remove(images_dir + filename)
            
            return redirect(url_for('upload_file',
                                    result=y_pred))
    return '''
    <!doctype html>
    
    <title>Upload a photo for analysis</title>
    <style>
        html{
            background-color: #99aaff;
            font-family: "Avenir";
            line-height: 10vw;
            text-align: center;
            
    </style>
    <h1>Upload JPEG for analysis</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    <h1 id="output"> %s </h1>
''' % result


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = False
    app.run()
