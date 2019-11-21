import numpy as np
from dataProcessors.Strategy import Strategy

class StrategyRandomClassRandomSample(Strategy):

    """Doesn't care about batch index.
    
    Returns:
        [type] -- [description]
    """

    def getBatch(self, generator, batchIndex):

        numClasses = generator.dataStats['countClasses']
        categoricalVals = np.eye(numClasses, dtype=np.float32)
        classIndices = np.random.randint(0, numClasses, generator.batch_size)
        choices = np.unique(classIndices, return_counts = True)

        freq = choices[1]
        X = np.empty((generator.batch_size, *generator.dim), dtype=np.float32)
        y = np.empty((generator.batch_size, numClasses), dtype=np.float32)
        
        # print( 'categorical values')
        # print(categoricalVals)
        sampleIndexInBatch = 0
        for classIndex in range(numClasses):
            numItems = freq[classIndex]
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
        
        print(f'getting data from low - {start} high - {end} of class {generator.classes[classIndex]}')
        choices = np.random.randint(start, end, size=size)

        for i in choices:
            chosenItems.append(allItems[i])

        
        return chosenItems

        







