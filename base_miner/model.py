import numpy as np 
import pandas as pd 
import kaggle
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from IPython.display import Image
import safetensors as st

img_height = 32
img_width = 32
batch_size = 500

def partition_datset():
    dataset_dir = "base_miner/data" # For Kaggle notebooks. If you run locally, point this line to the CIFAKE directory
    print("Loading dataset from: " + dataset_dir)

    # Load the training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir + "/train",
    seed = 512,
    image_size = (img_height, img_width),
    batch_size = batch_size)

    # Load the validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir + "/test",
    seed = 512,
    image_size = (img_height, img_width),
    batch_size = batch_size)

    print("Training Classes:")
    class_names = train_ds.class_names
    print(class_names)

    print("Testing Classes:")
    class_names = val_ds.class_names
    print(class_names)
    return train_ds, val_ds


# Function for plotting the error rate and metrics rate
def plot_metrics(history, metric):
    plt.plot(history.history[metric], label = metric)
    plt.plot(history.history['val_' + metric], label='val_' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()



train_ds, val_ds = partition_datset()    
# Constant values that will be shared by all the models
val_true_classes = np.concatenate([y for x, y in val_ds], axis = 0)  # Get true labels
class_names = ['FAKE', 'REAL']
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True) 


def build_resnet():
    # Building the Transfer Learning model using ResNet50
    ResNet_base_model = tf.keras.applications.ResNet50(
        include_top = False, 
        weights = 'imagenet', 
        input_shape = (img_height, img_width, 3),
        pooling = 'max'
    )
    ResNet_base_model.trainable = True

    # Create a new model on top of the ResNet50 base
    inputs = tf.keras.Input(shape = (img_height, img_width, 3))
    x = ResNet_base_model(inputs, training = False)
    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
    x = Dense(256, 
            kernel_regularizer = regularizers.l2(0.01), 
            activity_regularizer = regularizers.l1(0.01), 
            bias_regularizer = regularizers.l1(0.01),
            activation = 'relu')(x)
    x = Dropout(rate = .4, seed = 512)(x)       
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(1, activation = 'sigmoid')(x)
    ResNet_model = tf.keras.Model(inputs, outputs)

    # Compile the model
    ResNet_model.compile(
        optimizer = tf.keras.optimizers.Adamax(learning_rate = .001),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Summary of the model
    ResNet_model.summary()
    return ResNet_model

def build_vgg(): 
    # Building the Transfer Learning model using VGG16
    VGG_base_model = tf.keras.applications.VGG16(
        include_top = False, 
        weights = 'imagenet', 
        input_shape = (img_height, img_width, 3),
        pooling = 'max'
    )
    VGG_base_model.trainable = True

    # Create a new model on top of the VGG16 base
    inputs = tf.keras.Input(shape = (img_height, img_width, 3))
    x = VGG_base_model(inputs, training = False)
    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
    x = Dense(256, 
            kernel_regularizer = regularizers.l2(0.01), 
            activity_regularizer = regularizers.l1(0.01), 
            bias_regularizer = regularizers.l1(0.01),
            activation = 'relu')(x)
    x = Dropout(rate = .4, seed = 512)(x)       
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(1, activation = 'sigmoid')(x)
    VGG_model = tf.keras.Model(inputs, outputs)

    # Compile the Transfer Learning model
    VGG_model.compile(
        optimizer = tf.keras.optimizers.Adamax(learning_rate = .001),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Build the Transfer Learning model so we can see a summary
    VGG_model.summary()
    
def build_efficient_net():
    # Building the Transfer Learning model using EfficientNetV2B0
    EfficientNet_base_model = tf.keras.applications.EfficientNetV2B0(
        include_top = False, 
        weights = 'imagenet', 
        input_shape = (img_height, img_width, 3),
        pooling = 'max'
    )
    EfficientNet_base_model.trainable = True

    # Create a new model on top of the EfficientNet base
    inputs = tf.keras.Input(shape = (img_height, img_width, 3))
    x = EfficientNet_base_model(inputs, training = False)
    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
    x = Dense(256, 
            kernel_regularizer = regularizers.l2(0.01), 
            activity_regularizer = regularizers.l1(0.01), 
            bias_regularizer = regularizers.l1(0.01),
            activation = 'relu')(x)
    x = Dropout(rate = .4, seed = 512)(x)       
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(1, activation = 'sigmoid')(x)
    EfficientNet_model = tf.keras.Model(inputs, outputs)

    # Compile the Transfer Learning model
    EfficientNet_model.compile(
        optimizer = tf.keras.optimizers.Adamax(learning_rate = .001),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Build the Transfer Learning model so we can see a summary
    EfficientNet_model.summary()
    return EfficientNet_model

def train_model(model, model_name, train_ds, val_ds):
    print("Starting training with Transfer Learning using ResNet50...")
    model_history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = 200,
        verbose = 1,
        callbacks = [early_stopping]
    )
    print("Transfer Learning training finished.")
    
    model.save(f'{model_name}.h5')
    
    return model_history


if(__name__=='__main__'):
    train_ds, val_ds = partition_datset()    
    base_model = build_resnet()
    base_model_history = train_model(base_model, 'base_model', train_ds, val_ds)
    plot_metrics(base_model_history, 'loss')
    plot_metrics(base_model_history, 'accuracy')
    plot_metrics(base_model_history, 'precision')
    plot_metrics(base_model_history, 'recall')
    