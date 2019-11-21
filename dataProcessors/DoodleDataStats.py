import numpy as np
from os.path import dirname, basename, isfile
import glob, dill
import logging
from dataProcessors.DataUtils import DataUtils
from datetime import datetime


class DoodleDataStats:


    def __init__(self, folder='data/quickdraw-raw'):
        self.dataUtils = DataUtils()
        self.stats = {}
        pass


    def build(self, folder='data/quickdraw-raw'):

        self.stats['folder'] = folder
        self.stats['countClasses'] = 0
        self.stats['countItems'] = 0
        self.stats['classes'] = {}
        self.stats['maxPerClass'] = 0 # holds maximum number of items a class can have.
        files = glob.glob(folder + '/*.npy')

        # print(files)

        for f in files:
            self.addFileStats(f)
            self.stats['countClasses'] += 1
            if self.stats['countClasses'] % 10 == 0:
                logging.info('DoodleDataStats: processed ' + str(self.stats['countClasses']))
        pass


    def addFileStats(self, f):
        fNameNoExt = basename(f.replace("\\",'\/'))[:-4]
        className = self.dataUtils.convertFilenameToClass(fNameNoExt)
        data = np.load(f)

        self.stats['classes'][className] = data.shape[0]
        self.stats['countItems'] += data.shape[0]

        # check if this class has maximum number of items
        if data.shape[0] > self.stats['maxPerClass']:
            self.stats['maxPerClass'] = data.shape[0]

        pass

    
    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.stats, f)
        pass
        
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.stats = dill.load(f)
        pass

    
    def loadFromPersistentCacheByDate(self, savedOnDate):
        path = self.getPersistentPathFromDate(savedOnDate)
        self.load(path)
        return self.stats


    def saveToPersistenCacheWithToday(self):
        path = self.getPersistentPathFromDate()
        self.save(path)

    
    def getPersistentPathFromDate(self, savedOnDate = datetime.now()):
        # today = datetime.now()
        m = savedOnDate.strftime("%b")
        d = savedOnDate.strftime("%d")
        return  'persistentCache/doodleStats' + m + d + '.dill'