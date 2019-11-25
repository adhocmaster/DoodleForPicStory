import unittest
from dataProcessors.DoodleDataGeneratorByClass import DoodleDataGeneratorByClass
from dataProcessors.DoodleDataStats import DoodleDataStats
from datetime import datetime
import numpy as np


class DoodleDataGeneratorByClassTest(unittest.TestCase):

    def testBatches(self):

        dataStats = DoodleDataStats("folder")
        dataStats.loadFromPersistentCacheByDate(datetime(2019, 11, 24))

        
        generator = DoodleDataGeneratorByClass(dataStats.stats,split=0.7, part='first')

        # if self.part == 'first':
        #     self.n_batches =int( np.floor(self.dataStats['countItems'] * self.split) / self.batch_size) 
        # else:
        #     self.n_batches =int( np.floor(self.dataStats['countItems'] * (1 - self.split) ) / self.batch_size) 
        # pass

        assert generator.__len__() == int( np.floor(dataStats.stats['countItems'] * 0.7) / 16) 

        print(generator.n_batches)
        _, _ = generator.__getitem__(0)
        # print(X)
        # print(y)

        
        generator = DoodleDataGeneratorByClass(dataStats.stats,split=0.7, part='second', batchesPerEpoch=1000)
        assert generator.__len__() == 1000
        print(generator.n_batches)
        _, _ = generator.__getitem__(0)

        pass



if __name__ == '__main__':
    unittest.main()