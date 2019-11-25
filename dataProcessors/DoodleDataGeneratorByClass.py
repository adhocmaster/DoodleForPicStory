import numpy as np
import keras
from dataProcessors.DataUtils import DataUtils
from dataProcessors.GenerationStrategyType import GenerationStrategyType
from dataProcessors.ClassficationDataGenerator import ClassficationDataGenerator
from dataProcessors.StrategyRandomClassRandomSample import StrategyRandomClassRandomSample
from dataProcessors.StrategyPseudoRandomClassRandomSample import StrategyPseudoRandomClassRandomSample

import threading
	
def synchronized(func):
	
    func.__lock__ = threading.Lock()
		
    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func

class DoodleDataGeneratorByClass(ClassficationDataGenerator):
    'Generates data for Keras'
    def __init__(self, 
                dataStats:dict,
                split = 0.7,
                part = 'first',
                strategy:GenerationStrategyType = GenerationStrategyType.PseudoRandomClassRandomSample,
                batchesPerEpoch = None,
                batch_size:int = 16, 
                shuffle = True,
                maxCacheClasses = 10):

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
        self.maxCacheClasses = maxCacheClasses
        self.classCache = {}

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
        if self.strategy == GenerationStrategyType.PseudoRandomClassRandomSample:
            self.batchGenerator = StrategyPseudoRandomClassRandomSample()
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
    
    
    # Class data access
    def getClassItems(self, className):

        # print(f'Searching data for {className}')
        classData = self.getFromCache(className)
        if classData is not None:
            # print(f'Found {className} in cache')
            return classData

        path = self.dataStats['folder'] + '/' + self.dataUtils.convertClassToFilename(className) + '.npy'
        classData = np.load(path)
        self.saveInCache(className, classData)
        return classData

    
    def getFromCache(self, className):
        if className in self.classCache.keys():
            return self.classCache[className]
        return None

    @synchronized
    def saveInCache(self, className, classData):

        if className in self.classCache.keys():
            return
        if len(self.classCache) < self.maxCacheClasses:
            # print(f'Saving {className} in cache')
            self.classCache[className] = classData