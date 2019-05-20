import os
import numpy
import matplotlib.pyplot as plt
import random

from PIL import Image
from keras.preprocessing import image
from keras.applications import ResNet50
from keras import models, layers, callbacks, activations
from keras.backend import tf as ktf

train_images = []
train_images_labels = []
TRAIN_PATH = os.path.join("Caltech101", "training")
VALID_PATH = os.path.join("Caltech101", "validation")
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH))

# Creating generators from training and validation data
train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=32)

valid_datagen = image.ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=32)

# Freezing the ResNet50 layers
base_model = ResNet50(weights="imagenet")
for layer in base_model.layers:
	layer.trainable = False

X = base_model.layers[-2].output

#print("Hello")
"""
class LogEndResults(callbacks.Callback):
	def on_train_begin(self, logs):
		print(self.model)

result_logger = LogEndResults()

result_logger_2 = callbacks.LambdaCallback(on_train_end=lambda logs: print(logs))
"""
early_callback = callbacks.EarlyStopping(monitor="val_acc", patience=2, mode="auto")
#reduceLR_callback = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4)

import pandas as pd

try:
	log_df = pd.read_csv(os.path.join("AutoFC_ResNet", "AutoFC_ResNet_log.csv"), header=0)
except FileNotFoundError:
	log_df = pd.DataFrame(columns=["activation", "neurons", "dropout", "weight_initializer", "extra_layer_info", "time", "train_loss", "train_acc", "val_loss", "val_acc"])

print(log_df.shape)
#input()
"""
for activation in ["relu", "leaky", "tanh", "sigmoid"]:
	for neurons in (2 ** j for j in range(6, 13)):
		print("Model:", activation, neurons)
		X = layers.Dense(128, activation="relu")(X)
		X = layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(X)

		new_model = models.Model(inputs=base_model.input, outputs=X)
		new_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=["accuracy"])
		new_model.fit_generator(train_generator, validation_data=valid_generator, epochs=10, callbacks=[early_callback])
		FILE_NAME = f"{activation}{neurons}.h5"
		FILE_PATH = os.path.join("AutoFC_ResNet", "saved_models", FILE_NAME)
		print(f"Saving model {FILE_NAME}.")
		new_model.save(FILE_PATH)

		print(new_model.evaluate_generator(valid_generator, verbose=1))
		print(new_model.metrics_names)
"""

param_grid = {
	'activation': ['relu', 'tanh', 'sigmoid'],
	'neurons': (2  ** j for j in range(6, 13)),
	'dropout': numpy.arange(0, 0.99, 0.1),
	'weight_initializer': ['constant', 'normal', 'uniform', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'orthogonal', 'sparse'],
	'num_layers': range(1, 5)
}


from itertools import combinations,product
import time
import random

for hyper in product(*param_grid.values()):
	#temp_log_df = pd.DataFrame(list(columns=log_df.columns))
	start = time.time()

	#FILE_NAME = f"{activation}{neurons}.h5"
	#FILE_PATH = os.path.join("AutoFC_ResNet", "saved_models", FILE_NAME)
	#print("File name is ",FILE_NAME)
	#print("File path is ", FILE_PATH)
	
	num_layers = hyper[-1]
	inner_grid = {key: param_grid[key] for key in param_grid.keys() if key != 'num_layers'}
	inner_hyper = product(*inner_grid.values())
	
	for i in range(num_layers):
		random_sample = random.sample(list(inner_hyper), 1)
		print(random_sample)
		input()
		activation = i[0]
		neurons = i[1]
		dropout = i[2]
		weight_init = i[3]

		print("Model:", activation, neurons, dropout, weight_init)
		X = layers.Dense(neurons, activation=activation, kernel_initializer=weight_init)(X)
		X = layers.Dropout(dropout)(X)
	X = layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(X)
	
	new_model = models.Model(inputs=base_model.input, outputs=X)
	new_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=["accuracy"])
	history = new_model.fit_generator(train_generator, validation_data=valid_generator, epochs=10, callbacks=[early_callback])
	#print(f"Saving model {FILE_NAME}.")
	#new_model.save(FILE_PATH)

	time_taken = time.time() - start
	print(new_model.evaluate_generator(valid_generator, verbose=1))
	print(new_model.metrics_names)


	print("Time:", time_taken)

	# log the reults in the log dataframe
	loss, acc = new_model.evaluate_generator(train_generator)
	val_loss, val_acc = new_model.evaluate_generator(valid_generator)
	log_tuple = (activation, neurons, dropout, weight_init, time_taken, loss, acc, val_loss, val_acc)
	print("Logging results:", log_tuple)
	log_df.loc[log_df.shape[0]] = log_tuple

print(log_df.head())
