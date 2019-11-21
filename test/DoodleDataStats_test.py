import unittest
from dataProcessors.DoodleDataStats import DoodleDataStats
from dataProcessors.DataUtils import DataUtils
import datetime

class DoodleDataStatsTest(unittest.TestCase):

    def test_all(self):
        folder = 'data/quickdraw-raw'
        # filename = 'data/quickdraw-raw/full_numpy_bitmap_The Eiffel Tower.npy'
        dataStats = DoodleDataStats(folder)


        dataStats.build(folder)

        print(dataStats.stats)
        assert dataStats.stats['countClasses'] > 0

        assert 'The_Eiffel_Tower' in dataStats.stats['classes']

        dataStats.saveToPersistenCacheWithToday()
        pass


    def test_PersistentLoad(self):

        dataStats = DoodleDataStats("folder")

        dataStats.loadFromPersistentCacheByDate(datetime.datetime(2019, 11, 21))

        print(dataStats.stats)

        assert dataStats.stats['countClasses'] > 0

        assert 'The_Eiffel_Tower' in dataStats.stats['classes']



if __name__ == '__main__':
    unittest.main()