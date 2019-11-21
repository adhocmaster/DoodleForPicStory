import numpy as np
import keras
from dataProcessors.ClassficationDataGenerator import ClassficationDataGenerator

class DoodleDataGeneratorByClass(ClassficationDataGenerator):
    'Generates data for Keras'
    def __init__(self, dataStats, 
                batch_size=16, 
                shuffle=True):

        'Initialization'
        self.dim = (28, 28, 1)
        self.batch_size = batch_size
        self.n_channels = 1
        self.n_classes = dataStats['countclasses']
        self.n_batches =int(np.floor(len(dataStats['countItems']) / self.batch_size))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches 

    def __getitem__(self, batchIndex):
        'Generate one batch of data'

        

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)