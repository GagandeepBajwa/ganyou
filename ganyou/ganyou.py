'''
File Description: Main file
'''
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow.keras import layers

#defingin global variables
BUFFER_SIZE = 60000
BATCH_SIZE = 256

def train(generative_model, discriminative_model, training_dataset, epochs=50, noise_dim=100, num_examples_to_generate=16):
    print("Sarting training the model")

    #generating the seed that will be the input to the generative model and
    #   we will reuse the same seed over all the interations we want to have
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    #training code
    for epoch in range(epochs):
        start = time.time()

        #go over each bach of training data included
        for image_batch in training_dataset:
            train_step(image_batch)
        
        #generate the image after specific number of epochs
        
        

@tf.function
def train_step(images, generative_model, discriminative_model, noise_dim):
    #step in each iteration or what we call train step
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generative_model(noise, training=True)

      real_output = discriminative_model(images, training=True)
      fake_output = discriminative_model(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))




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

        #discrimative testing code goes here
        discriminative_model=build_discriminative_model()
        decision=discriminative_model(generated_image)
        print(decision)

        

def discriminator_loss(real_output, fake_output):
    #calculating the total loss of the fake output by the generative model
    #defining loss and optimizers
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    #calculating the generator loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

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

def generate_and_save_images_during_training(model, epch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


if __name__=="__main__":
    print("Welcome to GAN")
    
    #Calling fucntion to use GAN to genrate images
    generate_images()