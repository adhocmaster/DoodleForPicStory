import numpy as np
from dataProcessors.Strategy import Strategy

class StrategyRandomClassRandomSample(Strategy):

    """Doesn't care about batch index.
    
    Returns:
        [type] -- [description]
    """

    def __init__(self):
        self.stats = {}

    def getBatch(self, generator, batchIndex):

        numClasses = generator.dataStats['countClasses']
        categoricalVals = np.eye(numClasses, dtype=np.float32)
        
        # 1. Choose classes and number of items to fetch from each class
        choices = self.chooseClasses(numClasses, generator.batch_size)

        freq = choices[1]
        freqClassIndices = choices[0]
        X = np.empty((generator.batch_size, *generator.dim), dtype=np.float32)
        y = np.empty((generator.batch_size, numClasses), dtype=np.float32)
        
        # print( 'categorical values')
        # print(categoricalVals)

        # 2.0 fetch items for the chosen classes.
        sampleIndexInBatch = 0
        for i in range(len(freq)):
            classIndex = freqClassIndices[i]
            numItems = freq[i]

            # put into stats
            self.addToStats(classIndex, numItems)

            items = self.getRandomItems(generator, classIndex, numItems)

            for item in items:
                X[sampleIndexInBatch, ] = item.reshape(28, 28, 1) / 255
                y[sampleIndexInBatch, ] = categoricalVals[classIndex]
                sampleIndexInBatch += 1

        
        return X, y

    
    def getRandomItems(self,generator, classIndex, size):

        allItems = generator.getClassItems(generator.classes[classIndex])
        chosenItems = []

        end = allItems.shape[0]
        start = 0

        if generator.part == 'first':
            end = int( allItems.shape[0] * generator.split )
        else:
            start = int( allItems.shape[0] * generator.split )
        
        # print(f'getting data from low - {start} high - {end} of class {generator.classes[classIndex]}')
        choices = np.random.randint(start, end, size=size)

        for i in choices:
            chosenItems.append(allItems[i])

        
        return chosenItems

    
    def addToStats(self, classIndex, numItems):
        if classIndex in self.stats.keys():
            self.stats[classIndex] += numItems
        else:
            self.stats[classIndex] = numItems

    # override for different implementation.
    def chooseClasses(self, numClasses, batchSize):

        classIndices = np.random.randint(0, numClasses, batchSize)
        return np.unique(classIndices, return_counts = True)




