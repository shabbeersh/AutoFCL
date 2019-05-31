import os
import numpy
import matplotlib.pyplot as plt
import random

from PIL import Image
from keras.preprocessing import image
from keras.applications import *
from keras import models, layers, callbacks, activations
from keras.backend import tf as ktf
#import keras.utils.Sequence
#from keras.utils import multi_gpu_model

train_images = []
train_images_labels = []
TRAIN_PATH = os.path.join("corel-1k", "training")
VALID_PATH = os.path.join("corel-1k", "validation")
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH))

# Creating generators from training and validation data
batch_size = 8
train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size)

valid_datagen = image.ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size)

# Freezing the InceptionV3 layers

#print("Hello")
"""
class LogEndResults(callbacks.Callback):
    def on_train_begin(self, logs):
        print(self.model)

result_logger = LogEndResults()

result_logger_2 = callbacks.LambdaCallback(on_train_end=lambda logs: print(logs))
"""
early_callback = callbacks.EarlyStopping(monitor="val_acc", patience=5, mode="auto")
#reduceLR_callback = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4)

import pandas as pd

try:
    log_df = pd.read_csv(os.path.join("AutoFC_InceptionV3", "AutoFC_InceptionV3_log_Corel_1k_random_search_v1.csv"), header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["index", "num_layers", "activation", "neurons", "dropout", "weight_initializer", "time", "train_loss", "train_acc", "val_loss", "val_acc"])
    log_df = log_df.set_index('index')


param_grid = {
    'activation': ['relu', 'tanh', 'sigmoid'],
    'neurons': (2  ** j for j in range(6, 11)),
    'dropout': numpy.arange(0, 0.5, 0.1),
    'weight_initializer': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal'],
    'num_layers': range(0, 4)
}


from itertools import combinations,product
import time

num_layers = param_grid['num_layers']
inner_grid = {key: param_grid[key] for key in param_grid.keys() if key != 'num_layers'}
inner_hyper = list(product(*inner_grid.values()))
#print(list(inner_hyper))
NUM_TOTAL_PARAMS = sum([len(list(param_grid[key])) for key in param_grid])
print(NUM_TOTAL_PARAMS)
for i in num_layers:

    temp_store = []
    NUMBER_OF_SAMPLES = 1 if i == 0 else 13
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

        base_model = InceptionV3(weights="imagenet")
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
        history = new_model.fit_generator(train_generator, validation_data=valid_generator, epochs=20, callbacks=[early_callback],steps_per_epoch=len(train_generator)/batch_size, validation_steps =len(valid_generator))
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
        log_df.to_csv(os.path.join("AutoFC_InceptionV3", "AutoFC_InceptionV3_log_Corel_1k_random_search_v1.csv"))
