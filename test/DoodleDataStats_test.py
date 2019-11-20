import unittest
from dataProcessors.DoodleDataStats import DoodleDataStats
from dataProcessors.DataUtils import DataUtils

class DoodleDataStatsTest(unittest.TestCase):

    def test_all(self):
        folder = 'data/quickdraw-raw'
        # filename = 'data/quickdraw-raw/full_numpy_bitmap_The Eiffel Tower.npy'
        dataStats = DoodleDataStats(folder)


        dataStats.build(folder)

        print(dataStats.stats)
        assert dataStats.stats['countClasses'] > 0

        assert 'The_Eiffel_Tower' in dataStats.stats['classes']

        savePath = 'persistentCache/doodleStatsNov20.dill'
        dataStats.save(savePath)

if __name__ == '__main__':
    unittest.main()