import os
import numpy
import matplotlib.pyplot as plt
import random
import keras
from PIL import Image
from keras.preprocessing import image
from keras.applications import NASNetMobile
from keras import models, layers, callbacks, activations
from keras.backend import tf as ktf
#import keras.utils.Sequence
from keras.utils import multi_gpu_model
import numpy as np
from keras.callbacks import ReduceLROnPlateau
DATA_FOLDER = "Oxford102Flowers"

train_images = []
train_images_labels = []
TRAIN_PATH = os.path.join(DATA_FOLDER, "training")
VALID_PATH = os.path.join(DATA_FOLDER, "validation")
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH))

# Creating generators from training and validation data
batch_size = 8
train_datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.nasnet.preprocess_input)
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size)

valid_datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.nasnet.preprocess_input)
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size)

# Freezing the ResNet50 layers

#print("Hello")
"""
class LogEndResults(callbacks.Callback):
    def on_train_begin(self, logs):
        print(self.model)

result_logger = LogEndResults()

result_logger_2 = callbacks.LambdaCallback(on_train_end=lambda logs: print(logs))
"""
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-10)

import pandas as pd

try:
    log_df = pd.read_csv(os.path.join("AutoFC_NASNet", "AutoFC_NASNet_log_Oxford102_random_search_v1.csv"), header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["index", "num_layers", "activation", "neurons", "dropout", "weight_initializer", "time", "train_loss", "train_acc", "val_loss", "val_acc"])
    log_df = log_df.set_index('index')


param_grid = {
    'activation': ['relu', 'tanh', 'sigmoid'],
    'neurons': (2  ** j for j in range(6, 11)),
    'dropout': numpy.arange(0, 0.6, 0.1),
    'weight_initializer': ['he_normal'],
    'num_layers': range(0, 3)
    #'weight_initializer': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal'],
}


from itertools import combinations,product
import time
#import random


    #temp_log_df = pd.DataFrame(list(columns=log_df.columns))

    #FILE_NAME = f"{activation}{neurons}.h5"
    #FILE_PATH = os.path.join("AutoFC_DenseNet", "saved_models", FILE_NAME)
    #print("File name is ",FILE_NAME)
    #print("File path is ", FILE_PATH)

num_layers = param_grid['num_layers']
inner_grid = {key: param_grid[key] for key in param_grid.keys() if key != 'num_layers'}
inner_hyper = list(product(*inner_grid.values()))
#print(list(inner_hyper))
NUM_TOTAL_PARAMS = sum([len(list(param_grid[key])) for key in param_grid])
print(NUM_TOTAL_PARAMS)
for i in num_layers:
    print("Hello Loop!")
    used_seq = []
    print("Hello 2")
    #we_need = list(product(*[inner_hyper for _ in range(i)]))

    print("Hello 3")
    #print("Population:", len(we_need), 20, 20 > len(we_need))
    #in_use = random.sample(we_need, 20)

    temp_store = []
    NUMBER_OF_SAMPLES = 1 if i == 0 else 33
    for z in range(NUMBER_OF_SAMPLES):
        use_now = random.sample(inner_hyper, i)
        while use_now in used_seq:
            use_now = random.sample(inner_hyper, i)

        temp_store.append(use_now)
    #used_seq.append(in_use)

    for j in temp_store:
        act_list = []
        neu_list = []
        drop_list = []
        weight_list = []

        base_model = NASNetMobile(weights="imagenet")
        #base_model.summary()
        for layer in base_model.layers:
            layer.trainable = False

        X = base_model.layers[-2].output

        for k in j:
            activation = k[0]
            neurons = k[1]
            dropout = k[2]
            weight_init = k[3]

            act_list.append(activation)
            neu_list.append(neurons)
            drop_list.append(dropout)
            weight_list.append(weight_init)



            print("Model:", i, activation, neurons, dropout, weight_init)
            X = layers.Dense(neurons, activation=activation, kernel_initializer=weight_init)(X)
            X = layers.Dropout(dropout)(X)
            X = layers.BatchNormalization()(X)
        X = layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(X)

        new_model = models.Model(inputs=base_model.input, outputs=X)
        new_model = multi_gpu_model(new_model, gpus=2)
        new_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=["accuracy"])
        start = time.time()
        history = new_model.fit_generator(train_generator, validation_data=valid_generator, epochs=20, callbacks=[lr_reducer],steps_per_epoch=len(train_generator)/batch_size, validation_steps =len(valid_generator))
    #print(f"Saving model {FILE_NAME}.")
    #new_model.save(FILE_PATH)
        time_taken = time.time() - start
        print("Time:", time_taken)

    # log the reults in the log dataframe
        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))

        log_tuple = (i, act_list, neu_list, drop_list, weight_list, time_taken, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
        #log_tuple = (i, act_list, neu_list, drop_list, weight_list, time_taken, loss, acc, val_loss, val_acc)
        print("Columns:", log_df.columns)
        print("Logging results:", log_tuple)
        log_df.loc[log_df.shape[0]] = log_tuple
        print(log_df.shape)

        if log_df.shape[0] <= 5:
            print(log_df.head())

        print("Shape:", log_df.shape)

#print(log_df.head())
        log_df.to_csv(os.path.join("AutoFC_NASNet", "AutoFC_NASNet_log_Oxford102_random_search_v1.csv"))
