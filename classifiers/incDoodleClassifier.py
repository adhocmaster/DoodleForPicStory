from datetime import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

##########Live Plots ###################
if sys.modules.get( 'library.LivePlotKeras', False ) != False :
    del sys.modules['library.LivePlotKeras'] 
if sys.modules.get( 'LivePlotKeras', False ) != False :
    del sys.modules['LivePlotKeras'] 
from library.LivePlotKeras import *

logging.warning( "LivePlotKeras loaded" )

livePlotKeras = LivePlotKeras('categorical_accuracy')

############# Data Stats ##################
##### Expects dataStatsDate to load persistent data stats ##############
if sys.modules.get( 'dataProcessors.DoodleDataStats', False ) != False :
    del sys.modules['dataProcessors.DoodleDataStats'] 
if sys.modules.get( 'DoodleDataStats', False ) != False :
    del sys.modules['DoodleDataStats'] 
    
from dataProcessors.DoodleDataStats import DoodleDataStats
dataStats = DoodleDataStats("folder")
dataStats.loadFromPersistentCacheByDate(dataStatsDate)


################# Generators ##########################
if sys.modules.get( 'dataProcessors.DoodleDataGeneratorByClass', False ) != False :
    del sys.modules['dataProcessors.DoodleDataGeneratorByClass'] 
if sys.modules.get( 'DoodleDataGeneratorByClass', False ) != False :
    del sys.modules['DoodleDataGeneratorByClass'] 
    
from dataProcessors.DoodleDataGeneratorByClass import DoodleDataGeneratorByClass

trainGenerator = DoodleDataGeneratorByClass(dataStats.stats,split=0.7, part='first', batch_size = batch_size, batchesPerEpoch = trainBatchesPerEpoch, maxCacheClasses=trainClassCache)
validationGenerator = DoodleDataGeneratorByClass(dataStats.stats,split=0.7, part='second', batch_size = batch_size, batchesPerEpoch = validationBatchesPerEpoch, maxCacheClasses=validationClassCache)

################### Classifier Factory ############################
if sys.modules.get( 'classifiers.ClassifierFactory', False ) != False :
    del sys.modules['classifiers.ClassifierFactory'] 
if sys.modules.get( 'ClassifierFactory', False ) != False :
    del sys.modules['ClassifierFactory'] 
    
from classifiers.ClassifierFactory import ClassifierFactory
classifierFactory = ClassifierFactory()


################## Classifier IO ###########################
if sys.modules.get( 'classifiers.ClassifierIO', False ) != False :
    del sys.modules['classifiers.ClassifierIO'] 
if sys.modules.get( 'ClassifierIO', False ) != False :
    del sys.modules['ClassifierIO'] 
    
from classifiers.ClassifierIO import ClassifierIO
classifierIO = ClassifierIO()
