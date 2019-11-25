from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.models import load_model

class ClassifierFactory:


    def __init__(self):
        self.templates = [
            'UpSampling',
            'UpSamplingBN',
            'Basic',
            'BasicBN',
            'BasicSmallMaxPool'
        ]
        pass

    
    def create(self, 
        template, 
        outputClasses,
        inputShape = (28, 28, 1), 
        loss = losses.categorical_crossentropy,
        optimizer = optimizers.Nadam(lr=0.002),
        metrics = [metrics.categorical_accuracy]
    ):

        # if template not in self.templates:
        #     raise Exception(f'{template} does not have any template')

        method = getattr(self, 'get' + template, f'{template} does not have any template')
        modelInput = layers.Input(shape=inputShape)
        return method(outputClasses, modelInput, loss, optimizer, metrics)


    def getUpSampling(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics,
        batchNormalization = False 
        ):
        x = layers.Conv2D(64, 
            kernel_size = (5, 5), 
            padding = 'same'
            )(modelInput)

        x = layers.LeakyReLU(alpha=0.1)(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        else:
            x = layers.Dropout(0.1)(x)

        x = layers.Conv2D(64, kernel_size=(5, 5), activation=activations.tanh, padding='same')(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        x = layers.UpSampling2D(size=(2,2))(x)

        x = layers.Conv2D(8, kernel_size=(4, 4), activation=activations.relu, padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(outputClasses*10)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(outputClasses, activation=activations.softmax)(x)

        if batchNormalization:
            model = models.Model(modelInput, x, name = "Dropout-Max-Pool-UpSampling-BN")
        else:
            model = models.Model(modelInput, x, name = "Dropout-Max-Pool-UpSampling")
        # model.summary()
        model.compile(optimizer=optimizer,
             loss = loss,
             metrics = metrics)

        return model



    def getUpSamplingBN(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics 
        ):
        return self.getUpSampling(outputClasses, modelInput, loss, optimizer, metrics, batchNormalization=True)
        
    
    def getBasic(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics,
        batchNormalization = False 
        ):
        x = layers.Conv2D(64, 
            kernel_size = (5, 5), 
            padding = 'valid'
            )(modelInput)

        x = layers.ReLU()(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Conv2D(64, kernel_size=(5, 5), activation=activations.relu, padding='valid')(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(outputClasses*10)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(outputClasses, activation=activations.softmax)(x)

        if batchNormalization:
            model = models.Model(modelInput, x, name = "BasicBN")
        else:
            model = models.Model(modelInput, x, name = "Basic")
        # model.summary()
        model.compile(optimizer=optimizer,
             loss = loss,
             metrics = metrics)

        return model

        
    
    def getBasicBN(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics 
        ):
        return self.getBasic(outputClasses, modelInput, loss, optimizer, metrics, batchNormalization=True)


    
    
    def getBasicSmallMaxPool(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics,
        batchNormalization = False 
        ):
        x = layers.Conv2D(64, 
            kernel_size = (5, 5), 
            padding = 'same'
            )(modelInput)

        x = layers.ReLU()(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)

        x = layers.Conv2D(32, kernel_size=(5, 5), activation=activations.relu, padding='same')(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
        
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(outputClasses*2)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(outputClasses, activation=activations.softmax)(x)

        if batchNormalization:
            model = models.Model(modelInput, x, name = "BasicSmallMaxPoolBN")
        else:
            model = models.Model(modelInput, x, name = "BasicSmallMaxPool")
        # model.summary()
        model.compile(optimizer=optimizer,
             loss = loss,
             metrics = metrics)

        return model
