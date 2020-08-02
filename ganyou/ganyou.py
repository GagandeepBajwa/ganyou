'''
File Description: Main file
'''
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow.keras import layers


def train():
    print("Sarting training the model")

def generate_images(model_path=None):
    print("Starting generating the images")

    #get the genrative model
    if(model_path==None):
        #create a new generative model
        generative_model=build_generative_model()

        #create a new discrimator model
        #discriminative_model = build_discriminative_model()

        #code from here is just for testign purposes
        noise = tf.random.normal([1,100])
        generated_image = generative_model(noise, training=False)
        print(generated_image.shape)
        plt.imshow(generated_image[0,:,:,0], cmap="cubehelix")
        plt.show()


def build_generative_model():
    print("Starting building the model")
    
    #defining the model
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def build_discriminative_model():
    print("Starting building the discriminative model")

    #defining the model
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

if __name__=="__main__":
    print("Welcome to GAN")
    
    #Calling fucntion to use GAN to genrate images
    generate_images()