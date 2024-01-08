import tensorflow as tf
from tensorflow.keras import Sequential, layers
from adapter_tensorflow import Adaptertensorflow

class architecture:
    
    @staticmethod
    def cnn():

        model = model.Sequential()
        
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(factor=0.2))
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid'))

        return Adaptertensorflow(model)
