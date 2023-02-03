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
from src.util import *
import os

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_encoder(channels,latent_dim):
    """Create the encoder part of a VAE model
    
    Arguments:
        channels: Number of channels in the image; 3 for RGB
        latent_dim: Size of the latent vector (the middle part of the VAE
    """
    
    encoder_inputs = keras.Input(shape=(28, 28, channels))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return  keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
def create_decoder(channels,latent_dim):
    """Create the decoder part of a VAE model
    
    Arguments:
        channels: Number of channels in the image; 3 for RGB
        latent_dim: Size of the latent vector (the middle part of the VAE
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
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
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
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

def train_VAE(dataset,suffix,seed,save_location="", latent_dim=2,epochs=30):
    """Train a VAE model on some dataset, such as MNIST
    
    Arguments:
        dataset: Object from the dataset class, such as MNIST or CUB
        save_location: String that says where to store the VAE model; if empty it won't be stored
        latent_dim: Size of the latent dimension for the VAE model
        
    Returns: Nothing
    """
    
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    all_data = dataset.get_data(seed=seed,suffix=suffix)
    all_files = ['dataset'+i['img_path'] for i in all_data]
    images = np.array([file_to_numpy(i) for i in all_files])

    decoder_3 = create_decoder(3,latent_dim)
    encoder_3 = create_encoder(3,latent_dim)

    vae = VAE(encoder_3, decoder_3)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(images, epochs=epochs, batch_size=128)    
    
    if save_location != "":
        vae.save_weights("results/models/{}".format(save_location))

    save_vae(vae,dataset,suffix,seed)
        
def save_vae(model,dataset,suffix,seed):
    all_data = dataset.get_data()
    random.shuffle(all_data)
    
    all_data = all_data[:10000]
    all_images = [file_to_numpy("dataset/"+i['img_path']) for i in all_data]
    all_images = np.array(all_images)
    
    embeddings = model.encoder.predict(np.array(all_images).astype("float32")/255)[2]    

    average_embedding_by_concept = {}
    
    for index, name in enumerate(dataset.get_attributes()):
        embeddings_with_concept = np.array([embeddings[i] for i in range(len(all_data)) 
                                            if all_data[i]['attribute_label'][index] == 1])
        mean_embedding = np.mean(embeddings_with_concept,axis=0)
        average_embedding_by_concept[name] = np.array([mean_embedding])

    folder_name = "results/vae/{}/{}".format(dataset.experiment_name+suffix,seed)
        
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    for concept in average_embedding_by_concept:
        file_name = "{}/{}.npy".format(folder_name,concept)
        np.save(open(file_name,"wb"),average_embedding_by_concept[concept])

if __name___ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--dataset', type=str,
                        help='Name of the dataset which we generate, such as mnist')
    parser.add_argument('--suffix', type=str,
                        help='Specific subclass of dataset we\'re using')
    parser.add_argument('--seed',type=int, default=42,
                        help='Random seed used in tcav experiment')

    args = parser.parse_args()
    
    if args.dataset.lower() == 'mnist':
        dataset = MNIST_Dataset()
    elif args.dataset.lower() == 'cub':
        dataset = CUB_Dataset()
    else:
        raise Exception("{} not implemented".format(args.dataset))
    
    seed = args.seed
    suffix = args.suffix
    
    if args.algorithm == 'vae':
        train_VAE(dataset,suffix,seed)
