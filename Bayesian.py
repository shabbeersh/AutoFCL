import os
import numpy
import matplotlib.pyplot as plt
import random

from PIL import Image
from keras.preprocessing import image
#from keras.applications import ResNet50
#from keras import models, layers, callbacks, activations
#from keras.backend import tf as ktf
#from keras.utils import multi_gpu_model, Sequence
from bayes_opt import BayesianOptimization

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
import pandas as pd

TRAIN_PATH = os.path.join("Caltech101", "training")
VALID_PATH = os.path.join("Caltech101", "validation")
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH))

# Creating generators from training and validation data
train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=32)

valid_datagen = image.ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=32)

try:
	log_df = pd.read_csv(os.path.join("AutoFC_ResNet", "AutoFC_ResNet_log_Bayesian.csv"), header=0, index_col=['index'])
except FileNotFoundError:
	log_df = pd.DataFrame(columns=["index", "num_layers", "activation", "neurons", "dropout", "weight_initializer", "time", "train_loss", "train_acc", "val_loss", "val_acc"])
	log_df = log_df.set_index('index')


param_grid = {
	'activation': ['relu', 'tanh', 'sigmoid'],
	'neurons': (2  ** j for j in range(6, 13)),
	'dropout': numpy.arange(0, 0.99, 0.1),
	'weight_initializer': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal'],
	'num_layers': range(1, 5)
}

"""
class BatchGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_sete
        self.batch_size = batch_size

    def __len__(self):
        return int(numpy.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return numpy.array([resize(imread(file_name), (224, 224)) for file_name in batch_x]), np.array(batch_y)


train_generator = BatchGenerator()
"""

def get_model(num_layers, num_neurons, dropout):
    base_model = ResNet50(weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False

    X = base_model.layers[-2].output

    for _ in range(num_layers):
        X = layers.Dense(num_neurons, activation='relu')(X)
        X = layers.Dropout(dropout)(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(X)
    model = models.Model(inputs=base_model.inputs, outputs=X)
    return model


def model_fit(num_layers, num_neurons, dropout):
	model = get_model(num_layers, num_neurons, dropout)
	model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit_generator(train_generator, epochs=2, validation_data=valid_generator, verbose=0)
	score = model.evaluate_generator(valid_generator, verbose=0)
	return score[1]


from functools import partial

model_partial = partial(model_fit, num_layers=2, num_neurons=32)

pbounds = {
	#'num_layers': (1, 4),
	#'activation': ['relu', 'tanh', 'sigmoid'],
	'dropout': (0, 0.99),
	#'num_neurons': (32, 128)
}

optimizer = BayesianOptimization(
	f=model_partial,
	pbounds=pbounds,
	verbose=2
)

optimizer.maximize(init_points=10, n_iter=10)

for i, res in enumerate(optimizer.res):
	print('Iteration: {}\n\t{}'.format(i, res))

print(optimizer.max)
