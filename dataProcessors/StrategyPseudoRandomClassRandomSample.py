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

        perclass = int(batchSize / maxChoices)

        newFreq = np.full(maxChoices, perclass, dtype=int)
        newFreqClassIndices = np.random.choice(freqClassIndices, maxChoices)
        newFreq[0] = perclass + batchSize - perclass*maxChoices
        
        return (newFreqClassIndices, newFreq)