import os
import numpy
import matplotlib.pyplot as plt
import random

from PIL import Image
from keras.preprocessing import image
from keras.applications import *
from keras import models, layers, callbacks, activations
from keras.backend import tf as ktf
from keras.utils import multi_gpu_model, Sequence
from keras.callbacks import ReduceLROnPlateau
from keras.utils import multi_gpu_model
from datetime import datetime

import pandas as pd

import GPyOpt, GPy
early_callback = callbacks.EarlyStopping(monitor="val_acc", patience=5, mode="auto")
batch_size=8
TRAIN_PATH = os.path.join("Caltech101", "training")
VALID_PATH = os.path.join("Caltech101", "validation")
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH))

# Creating generators from training and validation data
train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=8)

valid_datagen = image.ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=8)

def get_model(num_layers, num_neurons, dropout, activation, weight_initializer):
    base_model = VGG16(weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False

    X = base_model.layers[-4].output
    # change
    for i in range(num_layers):
        X = layers.Dense(num_neurons[i], activation=activation, kernel_initializer=weight_initializer)(X)
        X = layers.Dropout(dropout[i])(X)
        X = layers.BatchNormalization()(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(X)
    model = models.Model(inputs=base_model.inputs, outputs=X)
    return model

try:
    log_df = pd.read_csv(os.path.join("AutoFC_VGG16", "AutoFC_VGG16_log_CalTech_101_bayes_opt_v1.csv"), header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'dropout', 'num_neurons', 'num_layers', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')

print("Shape:", log_df.shape)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-10)
from itertools import product
p_space = {
    'activation': ['relu', 'tanh', 'sigmoid'],
    'weight_initializer': ['he_normal'],
    'num_layers': list(range(0,3))
    #'weight_initializer': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal']
}
p_space = list(product(*p_space.values()))
start = datetime.time(datetime.now())
print("Starting:", start)
for combo in p_space:
    print(combo)
    activation, weight_initializer, num_layers = combo
    bounds = []
        #{'name': 'num_neurons', 'type': 'discrete', 'domain': [2 ** j for j in range(6, 11)]},
        #{'name': 'num_layers', 'type': 'discrete', 'domain': range(0, 4)}
        #{'name': 'activation', 'type': 'discrete', 'domain': ['relu', 'tanh', 'sigmoid']},
        #{'name': 'weight_initializer', 'type': 'discrete', 'domain': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal']}
    
    for i in range(num_layers):
        bounds.append({'name': 'dropout' + str(i + 1), 'type': 'discrete', 'domain': numpy.arange(0, 0.6, 0.1)})
    for i in range(num_layers):
        bounds.append({'name': 'num_neurons' + str(i + 1), 'type': 'discrete', 'domain': [2 ** j for j in range(6, 11)]})

    history = None
    neurons = None
    dropouts = None
    def model_fit(x):
        global neurons
        global dropouts
        dropouts = [int(x[:, i]) for i in range(0, num_layers)] 
        neurons = [int(x[:, i]) for i in range(num_layers, len(bounds))]
        print("Current Parameters:")
        # for i in range(num_layers):
        #     print("\t{}:\t{}".format(bounds[i]['name'], x[:, 0]))
        #     print("\t{}:\t{}".format(bounds[i + 1]['name'], x[:, i + 1]))
        model = get_model(
            dropout=dropouts,
            num_layers=num_layers,
            num_neurons= neurons,
            activation=activation,
            weight_initializer=weight_initializer
        )
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        global history
        history = model.fit_generator(train_generator, validation_data=valid_generator, epochs=40, callbacks=[lr_reducer],steps_per_epoch=len(train_generator)/batch_size, validation_steps =len(valid_generator))
        #score = model.evaluate_generator(valid_generator, verbose=1)
        return min(history.history['val_loss'])
    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit, domain=bounds)
    opt_.run_optimization(max_iter=5)
    # print("""
    # Optimized Parameters:
    # \t{0}:\t{1}
    # \t{2}:\t{3}
    # \t{4}:\t{5}
    # """.format(bounds[0]["name"],opt_.x_opt[0],
    #            bounds[1]["name"],opt_.x_opt[1],
    #            bounds[2]["name"],opt_.x_opt[2],
    #            #bounds[3]["name"],opt_.x_opt[3],
    #            #bounds[4]["name"],opt_.x_opt[4],
    #            #bounds[5]["name"],opt_.x_opt[5]
    # ))
    print("Optimized Parameters:")
    for i in range(num_layers):
        print("\t{}:\t{}".format(bounds[i]['name'], opt_.x_opt[i]))

    for i in range(num_layers, len(bounds)):
        print("\t{}:\t{}".format(bounds[i]['name'], opt_.x_opt[i]))
    print("optimized loss: {0}".format(opt_.fx_opt))
    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    log_tuple = (activation, weight_initializer, dropouts, neurons, num_layers, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], opt_.fx_opt, history.history['val_acc'][best_acc_index])
    #print("Activation weight_initializer dropout_rate #neurons #FClayers train_loss train_acc val_loss val_acc")
    print("Logging record:", log_tuple)
    print('lof_df shape',log_df.shape[0])
    log_df.loc[log_df.shape[0]] = log_tuple
    print("Shape:", log_df.shape)

    log_df.to_csv(os.path.join("AutoFC_VGG16", "AutoFC_VGG16_log_CalTech_101_bayes_opt_v1.csv"))

end = datetime.time(datetime.now())
print("Ending:", end)
