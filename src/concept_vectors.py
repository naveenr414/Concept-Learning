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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier

def row_to_fix(logits,new_relations_0,new_relations_1,similar_0,similar_1):
    x_vals = []
    
    for i in range(len(logits)):
        logit_value = logits[i]
        similar_0 = new_relations_0[i]
        similar_probs = [k[1] for k in similar_0]
        similar_0_logits = [logits[k[0]] for k in similar_0]
        
        
        similar_1 = new_relations_1[i]
        similar_probs_1 = [k[1] for k in similar_1]
        similar_1_logits = [logits[k[0]] for k in similar_1]
        
        temp = [logit_value]
        temp += similar_probs
        temp += similar_0_logits
        temp += similar_probs_1
        temp += similar_1_logits
                
        x_vals.append(temp)
        
    return x_vals

def row_to_fix_2(logits,new_relations_0,new_relations_1,similar_0,similar_1):
    x_vals = []
    
    for i in range(len(logits)):
        logit_value = logits[i]
        similar_0 = new_relations_0[i]
        similar_probs = [k[1] for k in similar_0]
        similar_0_logits = [logits[k[0]] for k in similar_0]
        
        
        similar_1 = new_relations_1[i]
        similar_probs_1 = [k[1] for k in similar_1]
        similar_1_logits = [logits[k[0]] for k in similar_1]
        
        temp = [logit_value]
        temp += similar_probs
        temp += similar_0_logits
        temp += similar_probs_1
        temp += similar_1_logits
                
        x_vals.append(temp)
        
    return x_vals

def get_fix_data(logits,x,new_relations_0,new_relations_1,similar_0,similar_1,func=row_to_fix):
    fix = []
    y = []
    
    for row in logits:
        fix += func(row,new_relations_0,new_relations_1,similar_0,similar_1)
    
    for row in x:
        y += list(row)
    fix = np.array(fix)
    y = np.array(y)
    
    return fix, y

def get_new_relatations(train_x):
    new_relations_0 = {}
    new_relations_1 = {}

    for i in range(num_columns):
        # See what makes this 0
        all_nums = []

        for j in range(num_columns):
            if i != j:
                times_when_i_0 = 1-train_x[:,i]
                times_when_j_0 = 1-train_x[:,j]
                total_i_0 = len((times_when_i_0).nonzero()[0])
                total_j_0 = len((times_when_j_0).nonzero()[0])
                frac_times = len(((times_when_i_0)*(times_when_j_0)).nonzero()[0])

                if total_j_0 != 0:
                    frac_times /= total_j_0
                else:
                    frac_times = 0
                all_nums.append((j,frac_times))

        new_relations_0[i] = sorted(all_nums,key=lambda k: k[1],reverse=True)[:5]
    for i in range(num_columns):
        # See what makes this 0
        all_nums = []

        for j in range(num_columns):
            if i != j:
                times_when_i_1 = train_x[:,i]
                times_when_j_1 = train_x[:,j]
                total_i_1 = len((times_when_i_1).nonzero()[0])
                total_j_1 = len((times_when_j_1).nonzero()[0])
                frac_times = len(((times_when_i_1)*(times_when_j_1)).nonzero()[0])

                if total_j_1 != 0:
                    frac_times /= total_j_1
                else:
                    frac_times = 0
                all_nums.append((j,frac_times))

        new_relations_1[i] = sorted(all_nums,key=lambda k: k[1],reverse=True)[:5]
        
    return new_relations_0, new_relations_1

def fix_predictions(method,dataset,seed
                    train_logits_file,valid_logits_file,test_logits_file,
                    train_preprocessed_file,valid_preprocessed_file,test_preprocessed_file,
                    output_train,output_valid,output_test):
    """
    Arguments:
        method: Function such as load_shapley_vectors_Simple
        dataset: Object from the Dataset class
        seed: Number such as 43
        
        train_logits_file: String with npy file containing input train logits (such as train_logits.npy)
        valid_logits_file: String with npy file containing input valid logits (such as valid_logits.npy)
        test_logits_file: String with npy file containing input test logits (such as test_logits.npy)
        
        train_preprocessed_file: String with .pkl file containing train preprocessed (such as train.pkl)
        valid_preprocessed_file: String with .pkl file containing valid preprocessed (such as valid.pkl)
        test_preprocessed_file: String with .pkl file containing test preprocessed (such as test.pkl)
        
        output_train: String with npy file on where to dump results (such as train_logits_fixed.npy)
        output_valid: String with npy file on where to dump results (such as valid_logits_fixed.npy)
        output_test: String with npy file on where to dump results (such as test_logits_fixed.npy)
        
    Returns: Nothing
    
    Side Effects: Writes the output npy files for train, valid, test"""
    
    embedding_matrix = np.array([method(a,dataset,"",seed)[0] for a in dataset.get_attributes()])
    sim_matrix = cosine_similarity(embedding_matrix)
    clf_concept = MLPClassifier(max_iter=100)
    
    train_logits = np.load(train_logits_file)
    valid_logits = np.load(valid_logits_file)
    test_logits = np.load(test_logits_file)
    
    new_relations_0, new_relations_1 = get_new_relations(train_x)
        
    train_fix_concept, train_fix_y = get_fix_data(train_logits,train_x,
                                                  new_relations_0,new_relations_1,func=row_to_fix_2)
    valid_fix_concept, valid_fix_y = get_fix_data(valid_logits,valid_x,
                                                  new_relations_0,new_relations_1,func=row_to_fix_2)
    test_fix_concept, test_fix_y = get_fix_data(test_logits,test_x,
                                                new_relations_0,new_relations_1,func=row_to_fix_2)
    
    clf_concept.fit(train_fix_concept,train_fix_y)
    train_predict_concept_proba = clf_concept.predict_proba(train_fix_concept)[:,1].reshape(train_x.shape)
    valid_predict_concept_proba = clf_concept.predict_proba(valid_fix_concept)[:,1].reshape(valid_x.shape)
    test_predict_concept_proba = clf_concept.predict_proba(test_fix_concept)[:,1].reshape(test_x.shape)
   
    np.save(open(output_train,"wb"),train_predict_concept_proba)
    np.save(open(output_valid,"wb"),valid_predict_concept_proba)
    np.save(open(output_test,"wb"),test_predict_concept_proba)

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
    
    if 'chexpert' in experiment_name or 'dsprites' in experiment_name:
        experiment_name = experiment_name.split("_")[0]
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
            if len(all_matching_files) == 0:
                print(concept)
                                        
            for file_name in all_matching_files:
                temp_concept = concept.replace("(","\(").replace(")","\)")
                re_search = re.search('{}-random(.*)-{}-linear-{}.pkl'.format(temp_concept,bottleneck,alpha),file_name)
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

def create_vector_from_label(attribute_name,dataset,suffix,seed=-1):
    """Generate sparse concept vectors, by looking at whether a concept is present in a data point
        This produces a 0-1 vector, with the vector <0,1,0> representing
        the presence of the attribute in data point 1, and not present in datapoints 0, 2
        
    Arguments:
        attribute_name: String representing one of the attributes
        dataset: Object from the Dataset class

    Returns:
        concept_vector: Numpy vector representing the concept vector for the attribute
    """
        
    all_attributes = dataset.get_attributes()
    if attribute_name not in all_attributes:
        raise Exception("Unable to generate vector from attribute {}".format(attribute_name))
    index = all_attributes.index(attribute_name)
    
    train_data = dataset.get_data(suffix=suffix,seed=seed)
    concept_vector = [i['attribute_label'][index] for i in train_data]
    return np.array(concept_vector).reshape((1,len(concept_vector)))

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
    cem_vectors = load_cem_vectors(dataset.experiment_name+suffix,attribute_index,seed)
    if len(cem_vectors) != 0:
        return cem_vectors
    else:
        return np.zeros((1,cem_vectors.shape[1]))
    
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
    
def load_random_vectors_simple(attribute,dataset,suffix,seed=-1):
    return np.random.random((1,100))
    
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
    
def export_concept_vector(embedding_method,name,dataset,seeds):
    """Export concept vectors into a numpy file for use with the CEM intervention experiments
    
    Arguments:
        embedding_method: Function such as load_label_vectors_simple
        name: String representing the name of the method, used to name the output file
        dataset: Object from the dataset class
        seeds: List of numbers to loop over for the method

    Returns: Nothing
    
    Side Effects: Writes a npy file for each seed to the results/temp folder
    """
    
    for seed in seeds:
        all_embeddings =  np.array([np.mean(embedding_method(i,dataset,"",seed=seed),axis=0) for i in dataset.get_attributes()])
        np.save(open("results/temp/{}_{}.npy".format(name,seed),"wb"),all_embeddings)
