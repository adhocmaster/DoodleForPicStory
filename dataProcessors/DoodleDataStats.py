import numpy as np
from os.path import dirname, basename, isfile
import glob, dill
import logging
from dataProcessors.DataUtils import DataUtils


class DoodleDataStats:


    def __init__(self, folder='data/quickdraw-raw'):
        self.dataUtils = DataUtils()
        self.stats = {}
        # self.build(folder)
        pass


    def build(self, folder='data/quickdraw-raw'):

        self.stats['folder'] = folder
        self.stats['countClasses'] = 0
        self.stats['countItems'] = 0
        self.stats['classes'] = {}
        files = glob.glob(folder + '/*.npy')

        # print(files)

        for f in files:
            self.addFileStats(f)
            self.stats['countClasses'] += 1
            if self.stats['countClasses'] % 10 == 0:
                logging.info('DoodleDataStats: processed ' + str(self.stats['countClasses']))
        pass


    def addFileStats(self, f):
        fNameNoExt = basename(f.replace('\\','\/'))[:-4]
        className = self.dataUtils.convertFilenameToClass(fNameNoExt)
        data = np.load(f)
        self.stats['classes'][className] = data.shape[0]
        self.stats['countItems'] += data.shape[0]
        pass

    
    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.stats, f)
        pass
        
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.stats = dill.load(f)
        pass
