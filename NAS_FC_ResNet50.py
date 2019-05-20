from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten,Input
import time
import keras
from keras.optimizers import Adam
from keras import backend as K
import os,re
from keras.utils import np_utils
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from my_classes import DataGenerator
from glob import glob
#from sklearn.cross_validation import train_test_split
import os
import re
import pandas as pd
import numpy as np
import itertools

num_classes = 250
dataset1 = []
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

data_path = '/media/shabbeer/9256-40F0/Research_PhD/Datasets/ALOT Color full'
data_dir_list = os.listdir(data_path)


for dataset in sorted_alphanumeric(data_dir_list):
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    #print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        dataset1.append(img)

labels1 = np.ones((25000,), dtype='int16')
count1 = 0
for i in range(0,25000,100):
    for j in range(0,100):
        labels1[i+j] = count1
    count1 = count1 + 1


labels = dict(itertools.izip(dataset1,labels1))
#print(labels)


#from sklearn.model_selection import train_test_split
partition={}
x_train, x_test = list(train_test_split(dataset1, test_size=0.2))
print(type(x_train))
partition={'train':x_train,'validation':x_test}

#print(partition)

# Parameters
params = {'dim': (224,224),
          'batch_size': 32,
          'n_classes': 250,
          'n_channels': 3,
          'shuffle': True}

# Datasets
partition = partition
labels = labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
print(type(training_generator))
validation_generator = DataGenerator(partition['validation'], labels, **params)


image_input = Input(shape=(224, 224, 3))
epochs = 10



model = ResNet50(input_tensor=image_input, weights='imagenet',include_top=True)
model.summary()
last_layer = model.get_layer('flatten_1').output
#x= Flatten(name='flatten_1')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
NAS_resnet_FC = Model(image_input,out)
NAS_resnet_FC.summary()

for layer in NAS_resnet_FC.layers[:-1]:
	layer.trainable = False
print(NAS_resnet_FC.layers[-1].trainable)
NAS_resnet_FC.summary()


adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1.0e-6,amsgrad=False)
NAS_resnet_FC.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])


# Train model on dataset
history = NAS_resnet_FC.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,workers=6)

import matplotlib.pyplot as plt
print('Max Test accuracy:', history.history)

print('Max Test accuracy:', max(history.history['val_acc']))
# # visualizing losses and accuracy
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
