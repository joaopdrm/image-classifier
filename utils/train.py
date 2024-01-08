import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

def train(data_dir:str,validation_split:float,model,
          batch_size_:int,image_size:tuple):
    
    image_size_ = (image_size[0],image_size[1])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = validation_split,
        subset="training",
        seed=123,
        image_size = image_size_,
        batch_size = batch_size_
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size = image_size_,
    batch_size = batch_size_
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    model.fit(train_ds,val_ds)