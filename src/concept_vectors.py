import tcav
import numpy as np
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model as model
import tcav.tcav as tcav
import tcav.utils as utils
import glob
import pickle
import re
import argparse
from src.dataset import *
import random
import glob
import tensorflow as tf
from src.models import *
from src.util import *
import keras
import time
from sklearn.decomposition import PCA
import pandas as pd 

class ResnetWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])
    
class VGGWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])

def load_cem_vectors(experiment_name,concept_number,seed=-1,dataset_location="results/cem_concepts"):
    """Load all the 'active' embeddings from Concept Embedding Models
    
    Arguments:
        experiment_name: Which experiment these concepts come from
            Concept files are of form {experiment_name}_concept...
        concept_number: An index for which concept is used
            Should correspond to the index in the 'c' array in CEM
        loss: Whether or not to load the concept vectors with the extra 'concept_loss' term

    Returns:
        Numpy array of size (k,n), where k = number of concept vectors, and n = size of embedding
    """
    if seed == -1:
        all_seeds = get_seed_numbers(dataset_location+"/"+experiment_name)
        if len(all_seeds) == 0:
            raise Exception("No experiments found at "+dataset_location+"/{}/".format(experiment_name))
        
        seed = random.choice(all_seeds)
    
    file_location = dataset_location + "/{}/{}".format(experiment_name,seed)
    file_location+="/{}_concept_{}_active.npy".format(experiment_name,concept_number)
    return np.load(open(file_location,"rb"))

def load_tcav_vectors(concept,bottlenecks,experiment_name="unfiled",seed=-1,alphas=[0.1]):
    """Loads all the TCAV vectors for a given concept + bottleneck combination
    
    Arguments:
        concept: One ImageNet concept as a string
        bottlenecks: List of bottleneck layers in some model of study
        alpha (optional): Regularization parameter used during TCAV training

    Returns:
        Two things: 
            Numpy array of size (k,n), where k = number of concepts vectors, and n = size of the bottleneck
            List of size k, with metadata on each concept (the random comparison, alpha used, bottleneck)
    """
        
    if seed == -1:
        seed = get_seed_numbers("./results/cavs/{}".format(experiment_name))
        if len(seed) == 0:
            raise Exception("No CAVs found at {}".format("./results/cavs/{}".format(experiment_name)))
        seed = random.choice(seed)
    
    dataset_location = "./results/cavs/{}/{}".format(experiment_name,seed)
    all_matching_files = [] 
    concept_meta_info = []
    
    for bottleneck in bottlenecks:
        for alpha in alphas:
            file_name_pattern = "{}-random*-{}-linear-{}.pkl".format(concept,bottleneck,alpha)
            all_matching_files = glob.glob(dataset_location+"/"+file_name_pattern)
                                        
            for file_name in all_matching_files:
                re_search = re.search('{}-random(.*)-{}-linear-{}.pkl'.format(concept,bottleneck,alpha),file_name)
                random_concept = re_search.group(1)
                concept_meta_info.append({'alpha': alpha,'bottleneck':
                                          bottleneck, 'random_concept': int(random_concept), 'concept': concept})
                
    if len(concept_meta_info) == 0:
        raise Exception("No CAVs found at {}".format(dataset_location))

    concept_meta_info, all_matching_files = zip(*sorted(zip(concept_meta_info,all_matching_files), key=lambda k: k[0]['random_concept']))
                
    all_concept_vectors = []
    
    for file in all_matching_files:
        concept_vector = pickle.load(open(file,"rb"))['cavs'][0]
        all_concept_vectors.append(concept_vector)

    all_concept_vectors = np.array(all_concept_vectors)
    return all_concept_vectors, concept_meta_info

def load_tcav_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to load_tcav_vectors that is standardized across embeddings
    
    Arguments:
        attribute: Which TCAV concept we're looking to get vectors for, as a string
        dataset: Object from the datastet class
        suffix: Strnig representing which specific instance of the dataset we're testing 
    
    Returns: 
       Numpy array of TCAV vectors
    """
        
    return load_tcav_vectors(attribute,['block4_conv1'],experiment_name=dataset.experiment_name+suffix,seed=seed)[0]

def load_label_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to create_vector_from_label_cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
        suffix: String, representing which specific instance of the class we're testing 
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    vector = create_vector_from_label(attribute,dataset,suffix,seed=seed)
    
    return np.array(vector)
            
def load_cem_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to create vector from cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    all_attributes = dataset.get_attributes()
    attribute_index = all_attributes.index(attribute)
    return load_cem_vectors(dataset.experiment_name+suffix,attribute_index,seed)

def load_cem_loss_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to create vector from cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    all_attributes = dataset.get_attributes()
    attribute_index = all_attributes.index(attribute)
    return load_cem_vectors(dataset.experiment_name+suffix,attribute_index,seed,dataset_location="results/cem_concepts_loss")

def load_cem_stratified_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to create vector from cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    all_attributes = dataset.get_attributes()
    attribute_index = all_attributes.index(attribute)
    return load_cem_vectors(dataset.experiment_name+suffix,attribute_index,seed,dataset_location="results/cem_concepts_stratified")


def load_concept2vec_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Simplified call to create vector using a SkipGram that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of label-based vectors
    """

    
    dataset_location = "results/concept2vec"
    experiment_name = dataset.experiment_name+suffix
    
    if seed == -1:
        all_seeds = get_seed_numbers(dataset_location+"/"+experiment_name)
        
        if len(all_seeds) == 0:
            raise Exception("No experiments found at {}".format(dataset_location+"/"+experiment_name))
            
        seed = all_seeds[random.randint(0,len(all_seeds)-1)]
        
    all_vectors = np.load("{}/{}/vectors.npy".format(dataset_location+"/"+experiment_name,seed))
        
    all_attributes = dataset.get_attributes()
    attribute_index = all_attributes.index(attribute)
    
    vector = all_vectors[attribute_index,:]
    return vector.reshape((1,len(vector)))

def load_vector_from_folder(folder_name):
    def load_vectors_simple(attribute,dataset,suffix,seed=-1):
        return np.load(open("results/{}/{}/{}/{}.npy".format(folder_name,
                                                             dataset.experiment_name+suffix,seed,attribute),"rb"),allow_pickle=True)

    return load_vectors_simple
    
def load_model_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Develop a concept vector that's simply based on the model representation
    
    Arguments:
        attribute: A single attribute 
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of label-based vectors
    """

    return load_vector_from_folder("model_vectors")(attribute,dataset,suffix,seed=seed)
    
def load_vae_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Retrieve concept vectors based on a generative VAE model
    
    Arguments:
        attribute: A single attribute
        dataset: Object from the dataset class
        suffix: String, which is a specific instance of the dataset
        seed: Random seed
    
    Returns:
        Numpy array of latent representations from VAEs
    """

    return load_vector_from_folder("vae")(attribute,dataset,suffix,seed=seed)

def load_vae_concept_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Retrieve concept vectors based on a generative VAE model
    
    Arguments:
        attribute: A single attribute
        dataset: Object from the dataset class
        suffix: String, which is a specific instance of the dataset
        seed: Random seed
    
    Returns:
        Numpy array of latent representations from VAEs
    """

    return load_vector_from_folder("vae_concept")(attribute,dataset,suffix,seed=seed)
    
def load_shapley_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Retrieve concept vectors based on a generative VAE model
    
    Arguments:
        attribute: A single attribute
        dataset: Object from the dataset class
        suffix: String, which is a specific instance of the dataset
        seed: Random seed
    
    Returns:
        Numpy array of latent representations from VAEs
    """

    return load_vector_from_folder("shapley")(attribute,dataset,suffix,seed=seed)

    
def combine_embeddings_average(f_one,f_two):
    """Given two embedding functions, embedding_one and embedding_two, return a function that returns the 
        average embedding between the two functions
        
    Arguments:
        f_one: Function such as load_label_vectors_simple
        f_two: Function such as load_label_vectors_simple
        
    Returns: A function, similar to load_label_vectors_simple
    """
    
    def get_average_embedding(attribute,dataset,suffix,seed=-1):
        embedding_one = f_one(attribute,dataset,suffix,seed=seed)
        embedding_two = f_two(attribute,dataset,suffix,seed=seed)
        
        min_size = min(embedding_one.shape[1],embedding_two.shape[1])
        
        return np.mean(np.concatenate([embedding_one[:,:min_size],embedding_two[:,:min_size]]),axis=0).reshape((1,min_size))
    
    return get_average_embedding
    
def combine_embeddings_concatenate(f_one,f_two):
    """Given two embedding functions, embedding_one and embedding_two, return a function that returns the 
        concatenated embedding between the two functions
        
    Arguments:
        f_one: Function such as load_label_vectors_simple
        f_two: Function such as load_label_vectors_simple
        
    Returns: A function, similar to load_label_vectors_simple
    """
    
    def concatenate_per_row(A, B):
        """Concatenate every row in A to every row in B
        Taken from https://stackoverflow.com/questions/41589680/concatenation-of-every-row-combination-of-two-numpy-arrays
        """
        
        m1,n1 = A.shape
        m2,n2 = B.shape

        out = np.zeros((m1,m2,n1+n2),dtype=A.dtype)
        out[:,:,:n1] = A[:,None,:]
        out[:,:,n1:] = B
        return out.reshape(m1*m2,-1)

    def get_average_embedding(attribute,dataset,suffix,seed=-1):
        embedding_one = f_one(attribute,dataset,suffix,seed=seed)
        embedding_two = f_two(attribute,dataset,suffix,seed=seed)
        
        return concatenate_per_row(embedding_one,embedding_two)
    
    return get_average_embedding
    
    
