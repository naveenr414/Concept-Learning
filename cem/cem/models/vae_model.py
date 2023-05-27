import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import os
import argparse

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_encoder(size,channels,latent_dim):
    """Create the encoder part of a VAE model
    
    Arguments:
        channels: Number of channels in the image; 3 for RGB
        latent_dim: Size of the latent vector (the middle part of the VAE)
    """
    
    encoder_inputs = keras.Input(shape=(size,size, channels))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return  keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def create_encoder_model(channels,latent_dim,model_name="resnet50"):
    """Create the encoder part of a VAE model by using a pre-trained model
    
    Arguments:
        channels: Number of channels in the image; 3 for RGB
        latent_dim: Size of the latent vector (the middle part of the VAE)
        model_name: Which pre-trained model we're using, such as resnet50
        
    Returns: Keras model
    """
    
    if model_name == "resnet50":
        model = tf.keras.applications.resnet.ResNet50(weights = "imagenet", 
                               include_top=False, 
                               input_shape = (32,32,channels),
                               pooling='max')
    elif model_name == "vgg16":
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                include_top=False,
                                input_shape = (32,32,channels),
                                pooling='max')
   
    weights = np.array(model.get_weights())  
    new_weights = responsive_weights(weights)
    model.set_weights(new_weights)
    
    for l in model.layers:
        l.trainable = False
        
    x = tf.keras.layers.Flatten()(model.output)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return  keras.Model(model.input, [z_mean, z_log_var, z], name="encoder")
  
    
def create_decoder(size,channels,latent_dim):
    """Create the decoder part of a VAE model
    
    Arguments:
        channels: Number of channels in the image; 3 for RGB
        latent_dim: Size of the latent vector (the middle part of the VAE)
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(size//4 * size//4 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((size//4, size//4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder,concept_alignment=False,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.concept_alignment = concept_alignment
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.concept_loss_tracker = keras.metrics.Mean(name="concept_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.concept_loss_tracker,
        ]

    def train_step(self, data):
        if self.concept_alignment:
            data,concepts = data[0]
            concepts = tf.cast(concepts,tf.float32) 
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            concept_loss = 0
            
            if self.concept_alignment:
                masked_concepts = z*concepts
                reconstruction_mask = self.decoder(masked_concepts)
                concept_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction_mask), axis=(1,2)
                    )
                )
            
            total_loss = reconstruction_loss + kl_loss + concept_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.concept_loss_tracker.update_state(concept_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "concept_loss": self.concept_loss_tracker.result(),
        }

    
def create_skipgram_architecture(embedding_dimension,vocab_size,initial_embedding=None):
    """Create a skipgram architecture in keras, given an embedding dimension and a vocab size
    Do this by first turning a (target,context) pair into embeddings, taking the dot, then 
        enforcing that they're close to 1
    
    Arguments:
        embedding_dimension: Number representing the size of each embedding vector, such as 128
        vocab_size: Total number of words in the vocabulary; for concepts, total number of concepts
        
    Returns: Keras Model
    """
    
    # Target
    w_inputs = Input(shape=(1, ), dtype='int32')
    
    if type(initial_embedding) != type(None):
        w = Embedding(vocab_size, embedding_dimension, 
                     embeddings_initializer=tf.keras.initializers.Constant(initial_embedding))(w_inputs)
    else: 
        w = Embedding(vocab_size, embedding_dimension)(w_inputs)

    # Context 
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(vocab_size, embedding_dimension)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
    
    return SkipGram

def resize_cub(images):
    """Function to resize CUB images to 64x64 for the VAE
    
    Arguments:
        images: Numpy array of images that will be resized
        
    Returns: numpy array of the resized images
    """
    
    return np.array([tf.image.resize(i,(64,64)) for i in images])