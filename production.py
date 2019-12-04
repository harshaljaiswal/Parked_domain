#!/usr/bin/env python
#@author Harshaljaiswal

# impoorting essential libraries
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import shutil, os

# Loading the pretrained model on images.
model = load_model('model_saved.h5')

# summarize model.
# model.summary()
#{'parked': 0, 'unparked': 1}

# Acquiring test data
directory = "./test_data/"
files = glob.glob(directory+'*.png')

# prediction on test data or any data
def prediction(files):
    filename = []
    prediction = []
    for file in tqdm(files):
        test_image = image.load_img(file, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image, batch_size=1)
        filename.append(file)
        prediction.append(result[0][0])
    prediction = [round(x) for x in prediction]
    return(filename, prediction)


filename, prediction = prediction(files)
 
 # After we get the predictioins from model, create a dataframe which can be exported as csv files for further analysis.
df = pd.DataFrame(list(zip(filename, prediction)), columns =['filename', 'prediction'])
df.prediction.value_counts()
df.to_csv('prediction.csv', index= False)
df.prediction.value_counts()

# Determining if the domian is valid or parked from the prediction probabilities.
unparked = df[df['prediction']>0.6]
parked = df[df['prediction']<=0.6]

unparked_list = list(unparked.filename)
parked_list = list(parked.filename)

# segmenting the test data(websites) based on the prediction from model.
for f in unparked_list:
    shutil.copy(f, './unparked/')

for f in parked_list:
    shutil.copy(f, './parked/')
