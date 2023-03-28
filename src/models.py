import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dot
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from src.util import *
import os
import argparse
from src.dataset import *
import pandas as pd



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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

def train_VAE(dataset,suffix,seed,save_location="", latent_dim=2,epochs=30,concept_alignment=False):
    """Train a VAE model on some dataset, such as MNIST
    
    Arguments:
        dataset: Object from the dataset class, such as MNIST or CUB
        save_location: String that says where to store the VAE model; if empty it won't be stored
        latent_dim: Size of the latent dimension for the VAE model
        
    Returns: Nothing
    """
    
    if concept_alignment:
        latent_dim = len(dataset.get_attributes())
            
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    all_data = dataset.get_data(seed=seed,suffix=suffix)
    all_files = ['dataset/'+i['img_path'] for i in all_data]
    images = np.array([file_to_numpy(i) for i in all_files])
    concepts = np.array([i['attribute_label'] for i in all_data])
    
    size = 28
    
    if dataset.experiment_name == "cub":
        # Convert all images to the same size
        size = 64
        images = resize_cub(images)
        
    decoder_3 = create_decoder(size,3,latent_dim)
    encoder_3 = create_encoder(size,3,latent_dim)

    vae = VAE(encoder_3, decoder_3,concept_alignment=concept_alignment)
    vae.compile(optimizer=keras.optimizers.Adam())

    concepts = tf.convert_to_tensor(concepts)
    images = tf.convert_to_tensor(images)
    
    if concept_alignment:
        vae.fit([images,concepts], epochs=epochs, batch_size=128)    
    else:
        vae.fit(images,epochs=epochs,batch_size=128)
    
    if save_location != "":
        vae.save_weights("results/models/{}".format(save_location))

    save_vae(vae,dataset,suffix,seed,concept_alignment=concept_alignment)
        
def save_vae(model,dataset,suffix,seed,concept_alignment=False):
    all_data = dataset.get_data()
    random.shuffle(all_data)
    
    all_data = all_data[:10000] 
    all_images = [file_to_numpy("dataset/"+i['img_path']) for i in all_data]
    all_images = resize_cub(np.array(all_images))
    
    embeddings = model.encoder.predict(np.array(all_images).astype("float32")/255)[2]    

    average_embedding_by_concept = {}
    
    for index, name in enumerate(dataset.get_attributes()):
        embeddings_with_concept = np.array([embeddings[i] for i in range(len(all_data)) 
                                            if all_data[i]['attribute_label'][index] == 1])
        mean_embedding = np.mean(embeddings_with_concept,axis=0)
        average_embedding_by_concept[name] = np.array([mean_embedding])

    folder_name = "results/vae/{}/{}".format(dataset.experiment_name+suffix,seed)

    if concept_alignment:
        folder_name = "results/vae_concept/{}/{}".format(dataset.experiment_name+suffix,seed)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    for concept in average_embedding_by_concept:
        file_name = "{}/{}.npy".format(folder_name,concept)
        np.save(open(file_name,"wb"),average_embedding_by_concept[concept])
    
def get_large_image_model(dataset,model_name):
    """Create a VGG or other large image model, fine-tuned for our dataset
    
    Arguments:
        dataset: Object from the dataet class
        model: String, such as VGG16
        
    Returns: Keras Model
    """

    num_classes = len(set([i['class_label'] for i in dataset.get_data()]))
    if model_name.lower() == "vgg16":
        model_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise Exception("{} model not implemented yet".format(model_name))

    for layer in model_base.layers:
        layer.trainable = False
    
    fc1 = Flatten()(model_base.output)
    fc1 = Dense(1024, activation='relu')(fc1)
    output = Dense(num_classes, activation='softmax')(fc1)
    
    model = Model(inputs=model_base.input, outputs=output)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    return model
        
def get_large_image_model_concept(dataset,model_name):
    """Train a VGG model to predict concept values for a particular dataset
    
    Arguments:
        dataset: Object from dataset class
        model_name: String, such as 'VGG16'

    Returns: Keras model
    """
    
    num_attributes = len(dataset.get_data()[0]['attribute_label'])
    input_shape = (224, 224, 3)
    
    input_tensor = Input(shape=input_shape)

    if model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    else:
        raise Exception("{} model not implemented yet".format(model_name))
        
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_attributes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['mse'])
    return model
    
    
def train_large_image_model(dataset,model_name,suffix="",seed=42):
    """Train a VGG model to predict downstream task for some dataset
    
    Arguments:
        dataset: Object from the dataset class
        model_name: String, such as 'VGG16'
        suffix: String for the particular type of dataset
        seed: Random seed for model training
    
    Returns: Nothing
    
    Side Effects: Saves this model to results/models/vgg_models/...
    """
    
    model = get_large_image_model(dataset,model_name)
    
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    image_size = (224, 224)
    
    print("Seed {} suffix {}".format(seed,suffix))

    data = dataset.get_data(seed=seed,suffix=suffix)
    img_paths = ['dataset/'+i['img_path'] for i in data]
    labels = [str(i['class_label']) for i in data]
    train_df = pd.DataFrame(zip(img_paths,labels), columns=["image_path", "label"])

    np.random.seed(seed)
    random.seed(seed)
    
    train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                          x_col="image_path",
                                          y_col="label",
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          class_mode="categorical",
                                          shuffle=True)
    
    print("Generated dataset, now training model")

    model.fit(train_generator,
       steps_per_epoch=len(train_generator),
       epochs=25)

    model.save_weights("results/models/{}_models/{}_{}.h5".format(model_name.lower(),dataset.experiment_name+suffix,seed))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--suffix',type=str,help='Which subset of the dataset to use',
                        default='none')
    parser.add_argument('--dataset', type=str,
                        help='Name of the dataset which we generate, such as mnist')
    parser.add_argument('--seed',type=int, default=42,
                        help='Random seed used in tcav experiment')

    args = parser.parse_args()
    
    if args.dataset.lower() == "mnist":
        dataset = MNIST_Dataset()
    elif args.dataset.lower() == "cub":
        dataset = CUB_Dataset()
    else:
        raise Exception("{} not implemented".format(args.dataset))
    
    seed = args.seed
    suffix = args.suffix
    if suffix == 'none':
        suffix = '' 
    
    if args.algorithm == 'vae':
        train_VAE(dataset,suffix,seed,epochs=30,latent_dim=4)
    elif args.algorithm == 'vae_concept':
        train_VAE(dataset,suffix,seed,epochs=30,latent_dim=len(dataset.get_attributes()),concept_alignment=True)
    elif args.algorithm == 'shapley':
        train_large_image_model(dataset,'VGG16',suffix,seed=seed)