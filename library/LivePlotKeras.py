import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

import seaborn as sns
sns.set(style="darkgrid")

class LivePlotKeras(keras.callbacks.Callback):

    def __init__(self, metricName = 'mean_squared_error'):

        self.metricName = metricName
        self.valMetricName = 'val_' + self.metricName


    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.metricVals = []
        self.val_metricVals = []

        self.loss = []
        self.valLoss = []
        
        self.fig = plt.figure(figsize=(20, 10))
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.metricVals.append(logs.get(self.metricName))
        self.val_metricVals.append(logs.get(self.valMetricName))

        self.loss.append(logs.get('loss'))
        self.valLoss.append(logs.get('val_loss'))

        self.i += 1
        
        clear_output(wait=True)
        self.fig = plt.figure(figsize=(20, 10))
        plt.plot(self.x, self.loss, label="train Loss")
        plt.plot(self.x, self.valLoss, label="validation Loss")
        plt.plot(self.x, self.metricVals, label="train Accuracy")
        plt.plot(self.x, self.val_metricVals, label="validation Accuracy")
        plt.legend()
        plt.show()

        