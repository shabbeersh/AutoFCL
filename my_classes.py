import numpy as np
import keras
# import itertools,os,re


# num_classes = 250
# dataset1 = []
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(data, key=alphanum_key)
#
# data_path = '/media/shabbeer/9256-40F0/Research_PhD/Datasets/ALOT Color full'
# data_dir_list = os.listdir(data_path)
#
#
# for dataset in sorted_alphanumeric(data_dir_list):
#     img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
#     #print('Loaded the images of dataset-' + '{}\n'.format(dataset))
#     for img in img_list:
#         dataset1.append(img)
#
# labels1 = np.ones((25000,), dtype='int16')
# count1 = 0
# for i in range(0,25000,100):
#     for j in range(0,100):
#         labels1[i+j] = count1
#     count1 = count1 + 1
#
#
# labels = dict(itertools.izip(dataset1,labels1))
# #print(labels)
#
#
# params = {'dim': (224,224),
#           'batch_size': 32,
#           'n_classes': 250,
#           'n_channels': 3,
#           'shuffle': True}
#
#
# from sklearn.model_selection import train_test_split
# partition={}
# x_train, x_test = list(train_test_split(dataset1, test_size=0.2))


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=250, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]
        print(X, keras.utils.to_categorical(y, num_classes=self.n_classes))

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# training_generator = DataGenerator(x_train, labels, **params)
# print(training_generator.__)





