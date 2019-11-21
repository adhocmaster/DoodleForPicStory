import numpy as np
import keras
from dataProcessors.DataUtils import DataUtils
from dataProcessors.GenerationStrategyType import GenerationStrategyType
from dataProcessors.ClassficationDataGenerator import ClassficationDataGenerator
from dataProcessors.StrategyRandomClassRandomSample import StrategyRandomClassRandomSample

class DoodleDataGeneratorByClass(ClassficationDataGenerator):
    'Generates data for Keras'
    def __init__(self, 
                dataStats:dict,
                split = 0.7,
                part = 'first',
                strategy:GenerationStrategyType = GenerationStrategyType.RandomClassRandomSample,
                batchesPerEpoch = None,
                batch_size:int = 16, 
                shuffle = True):

        """[summary]
        """

        'Initialization'
        self.dataUtils = DataUtils()
        self.maxPerclass = dataStats['maxPerClass']
        self.dataStats = dataStats
        self.classes = list(dataStats['classes'].keys())
        self.split = split
        self.part = part # which part to generate after splitting
        self.strategy = strategy
        self.dim = (28, 28, 1)
        self.batch_size = batch_size
        self.batchGenerator = None # will be created by createBatchGenerator

        self.n_channels = 1
        self.n_classes = dataStats['countClasses']
        self.n_batches = batchesPerEpoch # will be calculated when calculateBatches is called
        self.shuffle = shuffle

        self.calculateBatches()
        self.createBatchGenerator()


        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches 



    def calculateBatches(self):

        if self.n_batches is not None: #user defined
            return 

        if self.part == 'first':
            # batches =int( np.floor(self.dataStats['countItems'] * self.split) / self.batch_size) 
            self.n_batches =int( np.floor(self.dataStats['countItems'] * self.split) / self.batch_size) 
        else:
            self.n_batches =int( np.floor(self.dataStats['countItems'] * (1 - self.split) ) / self.batch_size) 
        pass


    def createBatchGenerator(self):

        if self.strategy == GenerationStrategyType.RandomClassRandomSample:
            self.batchGenerator = StrategyRandomClassRandomSample()
            return

        raise Exception("Batch generator unavailable for " + self.strategy)


    def __getitem__(self, batchIndex):
        'Generate one batch of data'
        if self.batchGenerator is not None:
            return self.batchGenerator.getBatch(self, batchIndex)
        
        raise Exception("Batch generator unavailable for " + self.strategy)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
    
    
    def getClassItems(self, className):

        path = self.dataStats['folder'] + '/' + self.dataUtils.convertClassToFilename(className) + '.npy'
        return np.load(path)