import numpy as np
from dataProcessors.StrategyRandomClassRandomSample import StrategyRandomClassRandomSample


class StrategyPseudoRandomClassRandomSample(StrategyRandomClassRandomSample):

    """ It changes the distribution of population"""
    
    def chooseClasses(self, numClasses, batchSize):

        maxChoices = 3
        classIndices = np.random.randint(0, numClasses, batchSize)
        freqClassIndices, freq = np.unique(classIndices, return_counts = True)

        if len(freqClassIndices) <= maxChoices:
            return (classIndices, freq)

        newFreq = np.zeros(maxChoices, dtype=int)
        newFreqClassIndices = np.zeros(maxChoices, dtype=int)

        perclass = int(batchSize / maxChoices)
        for i in range(maxChoices):
            newFreqClassIndices[i] = freqClassIndices[i]
            newFreq[i] = perclass

        newFreq[i] = perclass + batchSize - perclass*maxChoices
        
        return (newFreqClassIndices, newFreq)