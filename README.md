# Radar-Sonar-GAN
Tensorflow scripts to train a GAN for generating synthetic radar/sonar data

This repository contains two files that define the same GAN architecture.  "radar_gan_keras.py" uses the Keras package 
within Tensorflow to define the network.  "radar_gan_low.py" uses low-level tensorflow operations and functions to define 
the same network architecture.

The generator takes an input vector of random gaussian noise of size (100,1) and generates a raw radar/sonar signal of size 
(1547520,2).  The generator architecture contains one fully connected layer, followed by three "deconvolutional" layers.
all using a leaky ReLU activation function except for the final convolutional layer, which has no activation function.

The discriminator's architecture contains three convolutional layers followed by two fully connected layers.  Each of these
layers also uses a leaky ReLU activation except for the output layer, which consists of a scalar value with a sigmoid 
activation function.

Both networks were adversarially trained using a stochastic approach and the AdamOptimizer method within Tensorflow.
The training process was executed on Googel Colab servers with GPU accelerated runtime.
