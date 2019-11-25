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
        x = layers.Dense(outputClasses*5)(x)
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
        batchNormalization = False,
        kernelSize = (5,5) 
        ):
        x = layers.Conv2D(64, 
            kernel_size = kernelSize, 
            padding = 'valid'
            )(modelInput)

        x = layers.ReLU()(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Conv2D(64, kernel_size=(3,3), activation=activations.relu, padding='valid')(x)
        if batchNormalization:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(outputClasses*5)(x)
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
    
    def getBasicSmallKernel(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics 
        ):
        return self.getBasic(outputClasses, modelInput, loss, optimizer, metrics, batchNormalization=True, kernelSize=(3,3))


    
    
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


    def getResNet(self, 
        outputClasses,
        modelInput, 
        loss,
        optimizer,
        metrics,
        batchNormalization = False 
        ):
        initialX = layers.Conv2D(64, 
            kernel_size = (6, 6), 
            padding = 'same'
            )(modelInput)

        initialX = layers.ReLU()(initialX)
        if batchNormalization:
            initialX = layers.BatchNormalization()(initialX)

        x = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), name='xPool')(initialX)
        x2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='x2Pool')(initialX)
        x3 = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='x3Pool')(initialX)

        # stage 1x

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same', name='xEntry')(x)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x = layers.Add(name='stage1x')([x, stage])

        # stage 2x

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x = layers.Add(name='stage2x')([x, stage])

        # stage 3x

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x = layers.Add(name='stage3x')([x, stage])

        
        # stage 1x2

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same', name='x2Entry')(x2)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x2 = layers.Add(name='stage1x2')([x2, stage])

        # stage 2x2

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x2)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x2 = layers.Add(name='stage2x2')([x2, stage])

        # stage 3x2

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x2)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x2 = layers.Add(name='stage3x2')([x2, stage])

        # stage 1x3

        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same', name='x3Entry')(x3)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x3 = layers.Add(name='stage1x3')([x3, stage])
        
        # stage 2x3
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x3)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x3 = layers.Add(name='stage2x3')([x3, stage])
        
        # stage 3x3
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(x3)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        stage = layers.Conv2D(64, kernel_size=(1, 1), activation=activations.relu, padding='same')(stage)
        if batchNormalization:
            stage = layers.BatchNormalization()(stage)
        x3 = layers.Add(name='stage3x3')([x3, stage])
        

        
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x2 = layers.MaxPool2D(pool_size=(2,2))(x2)
        x3 = layers.MaxPool2D(pool_size=(2,2))(x3)


        x = layers.Flatten(name='outputx')(x)
        x2 = layers.Flatten(name='outputx2')(x2)
        x3 = layers.Flatten(name='outputx3')(x3)

        mergedX = layers.concatenate(inputs=[x, x2, x3], name='mergedX')
        mergedX = layers.Dense(500)(mergedX)
        mergedX = layers.ReLU()(mergedX)
        mergedX = layers.Dropout(0.1)(mergedX)
        mergedX = layers.Dense(outputClasses, activation=activations.softmax, name='output')(mergedX)

        if batchNormalization:
            model = models.Model(modelInput, mergedX, name = "ResNetBN")
        else:
            model = models.Model(modelInput, mergedX, name = "ResNet")
        # model.summary()
        model.compile(optimizer=optimizer,
             loss = loss,
             metrics = metrics)

        return model