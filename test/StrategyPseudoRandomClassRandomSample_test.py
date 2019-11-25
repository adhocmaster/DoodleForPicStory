import unittest2
import numpy as np
from dataProcessors.StrategyPseudoRandomClassRandomSample import StrategyPseudoRandomClassRandomSample


class StrategyPseudoRandomClassRandomSampleTest(unittest2.TestCase):

    def testChooseClasses(self):
        strategy = StrategyPseudoRandomClassRandomSample()

        classes, freq = strategy.chooseClasses(100, 16)

        print('\n')
        print(classes)
        print(freq)
        
        assert np.sum(freq) == 16
        countSort = 0
        for i in range(1, len(classes)):
            if classes[i-1] <= classes[i]:
                countSort += 1
        
        assert countSort < 2
