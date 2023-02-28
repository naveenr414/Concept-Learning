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
                concept_meta_info.append({'alpha': alpha,'bottleneck': bottleneck, 'random_concept': int(random_concept), 'concept': concept})
                
    if len(concept_meta_info) == 0:
        raise Exception("No CAVs found at {}".format(dataset_location))

    concept_meta_info, all_matching_files = zip(*sorted(zip(concept_meta_info,all_matching_files), key=lambda k: k[0]['random_concept']))
                
    all_concept_vectors = []
    
    for file in all_matching_files:
        concept_vector = pickle.load(open(file,"rb"))['cavs'][0]
        all_concept_vectors.append(concept_vector)

    all_concept_vectors = np.array(all_concept_vectors)
    return all_concept_vectors, concept_meta_info

def create_concept2vec(dataset,suffix,seed=-1,
                             embedding_size=32,num_epochs=5,dataset_size=1000,initial_embedding=None):
    """Generate concept2vec vectors by training a Skipgram architecture on correlated concepts
    
    Arguments:
        dataset: Object from the Dataset Class
        suffix: String that represents a specific variant of the dataset
        seed: Number for the random seed
        embedding_size: Size of the embeddings; by default it's 32
        
    Returns: Nothing
    
    Side Effects: Trains a set of vectors at results/concept2vec/experimentname+suffix/seed/
    """
    
    if seed == -1:
        seed = random.randint(0,100000)
    
    destination_folder = "results/concept2vec/{}/{}".format(dataset.experiment_name+suffix,seed)
    
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    attributes = dataset.get_attributes()
    V = len(attributes)
    all_data = dataset.get_data(seed,suffix)[:dataset_size]
    
    SkipGram = create_skipgram_architecture(embedding_size,V,initial_embedding=initial_embedding)
    
    for _ in range(num_epochs):
        X = np.zeros((0,2),dtype=np.int32)
        Y = np.zeros((0,))

        # Collect the dataset
        for i, doc in enumerate(all_data):
            formatted_data = [attributes[j] for j,indicator in enumerate(doc['attribute_label']) if indicator == 1]
            data, labels = generate_skipgram_dataset(formatted_data,attributes,8,8)   
            
            data = np.array(data,dtype=np.int32)

            X = np.concatenate([X,data])
            Y = np.concatenate([Y,labels])
            
        X_0 = X[:,0]
        X_1 = X[:,1]
        
        SkipGram.fit([X_0,X_1],Y)
    
    vectors = SkipGram.get_weights()[0]
    
    np.save(open("{}/vectors.npy".format(destination_folder),"wb"),vectors)
    
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

def create_tcav_dataset(attribute_name,dataset,num_random_exp,max_examples=100,images_per_folder=50,seed=-1,suffix='',model_name="VGG16",bottlenecks=["block4_conv1"]):
    """Helper function to create TCAV from Attribute
        It creates the folder with images for the attribute, trains the TCAV vector,
        then deletes the folder
    
    Arguments:
        attribute_name: String containing one of the attributes for a dataset
        dataset: Object from the dataset class
        num_random_exp: How many random experiments to run
        max_examples: In the positive examples (attributes), how many images should there be 
        images_per_folder: In the random folders, how many images there should be 
        seed: Random seed

    Returns:
        None
        
    Side Effects:
        Trains a set of concept vectors, stored at ./results/cavs/experiment_name/seed
    """
    experiment_name = dataset.experiment_name+suffix
    
    if suffix == '_model_robustness':
        model_name = "VGG16Robustness"
        
    elif suffix == '_model_responsiveness':
        model_name = "VGG16Responsiveness"
        
    
    create_folder_from_attribute(attribute_name,
                                 dataset.get_images_with_attribute,num_images=max_examples,suffix=suffix,seed=seed)
    create_random_folder_without_attribute(
        attribute_name,num_random_exp,dataset.get_images_without_attribute,
        images_per_folder=images_per_folder,seed=seed,suffix=suffix)
    
    concepts = [attribute_name]
    target = "zebra"
    alphas = [0.1]
    
    create_tcav_vectors(concepts,target,model_name,bottlenecks,
                        num_random_exp,experiment_name=experiment_name,
                        alphas=[0.1],seed=seed,max_examples=max_examples)


def delete_previous_activations(bottleneck,attribute_list):
    """Delete all the previous activations so we can generate them 
    
    Arguments:
        bottleneck: String, such as mixed4c
        attribute_list: List of concepts which we want to delete and reset
        
    Returns: Nothing

    Side Effects: Deletes all files in activation_dirs corresponding to attributes in attribute_list
    """
    
    activation_dir = './results/activations'
    
    for concept in attribute_list:
        activation_file_location = "{}/acts_{}_{}".format(activation_dir,concept,bottleneck)
        if os.path.exists(activation_file_location):
            os.remove(activation_file_location)

    
def load_activations_model(experiment_name,max_examples,model_name,sess):
    """Given a model and an experiment name, create an activation class that loads activations
    
    Arguments:
        experiment_name: String representing which experiment we're running; this is a folder in 
            ./dataset/images
        max_examples: Maximum number of activations to load
        model_name: String representing the model name; currently we only support GoogleNet, VGG16, and Resnet50
        
    Returns:
        Object from ImageActivationGenerator
    """
    
    image_dir = "./dataset/images"
    activation_dir = './results/activations'
    models_used = ["GoogleNet",
                   "Resnet50","Resnet50Robustness","Resnet50Responsiveness",
                  "VGG16","VGG16Robustness","VGG16Responsiveness"]
    
    if model_name not in models_used:
        raise Exception("Model {} not implemented yet, select one of {}".format(model_name,models_used))
        
    if model_name == "GoogleNet":
        GRAPH_PATH = "./dataset/models/inception5h/tensorflow_inception_graph.pb"
        LABEL_PATH = "./dataset/models/inception5h/imagenet_comp_graph_label_strings.txt"
        mymodel = model.GoogleNetWrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
    elif "Resnet50" in model_name:
        if model_name == "Resnet50":
            GRAPH_PATH = "./dataset/models/keras/model_resnet.h5"
        LABEL_PATH = "./dataset/models/inception5h/imagenet_comp_graph_label_strings.txt"
        mymodel = ResnetWrapper(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
    elif "VGG16" in model_name:
        if model_name == "VGG16":
            GRAPH_PATH = './dataset/models/keras/model_vgg16.h5'
        elif model_name == "VGG16Robustness":
            GRAPH_PATH = './dataset/models/keras/model_vgg16_robust.h5'
        elif model_name == "VGG16Responsiveness":
            GRAPH_PATH = './dataset/models/keras/model_vgg16_responsive.h5'
            
        LABEL_PATH = "./dataset/models/inception5h/imagenet_comp_graph_label_strings.txt"
        mymodel = VGGWrapper(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
        
    act_generator = act_gen.ImageActivationGenerator(mymodel, image_dir, activation_dir,max_examples=max_examples)
    return act_generator
    
def get_activations_dictionary(attribute_list,sess,model_name="VGG16",
                               experiment_name="unfiled",max_examples=500,bottleneck="block4_conv1",delete_activations=True):
    """From a list of concepts or attributes, generate their representation in some model
        such as GoogleNet, at some bottleneck layer
        
    Arguments:
        attribute_list: String list of concepts to be analyzed, such as 'zebra'
        model_name: Which model to load these activations from
        experiment_name: String pointing to the folder where our images are
        max_examples: Maximum number of activations to load 

    Returns:
        Dictionary: Each key is an attribute, and each value is a numpy array of size 
            n x 100K, where n is the number of concept vectors, and 100K is the size of the 
            bottleneck
    """
    
    start = time.time()
    
    if model_name == "Resnet50":
        bottleneck = "conv4_block6_out"

    
    if delete_activations:
        delete_previous_activations(bottleneck,attribute_list)

    start = time.time()    
    act_generator = load_activations_model(experiment_name,max_examples,model_name,sess)
            
    acts = {}
    for i in attribute_list:
        start = time.time()
        examples = act_generator.get_examples_for_concept(i)
        activation_examples = act_generator.model.run_examples(examples, bottleneck)
        acts[i] = activation_examples
        shape = acts[i].shape
        acts[i] = acts[i].reshape((shape[0],shape[1]*shape[2]*shape[3]))
        
        
    return acts

def reset_tcav_vectors(concepts,num_random_exp,experiment_name,seed,bottleneck,alpha):
    """Delete all TCAV files in the CAV directory so we can re-train them 
    
    Arguments:
        concepts: List of attributes which we wish to delete
        num_random_exp: Number representing how many random experiments we want to delete from
        experiment_name: String for the name of the experiemnt we're deeleting from
        seed: Random seed, also a folder in the cav directory
        bottleneck: String for a layer in the model, such as mixed4c
        alpha: Hyperparameter, float 
        
    Returns: Nothing
    
    Side Effects: Deletes all CAVs in a folder
    """
    
    cav_dir = './results/cavs/{}/{}'.format(experiment_name,seed)
    
    for concept in concepts:
        for i in range(num_random_exp):
            tcav_file_location = "{}/{}-random500_{}-{}-linear-{}.pkl".format(cav_dir,concept,i,bottleneck,alpha)
            if os.path.exists(tcav_file_location):
                os.remove(tcav_file_location)
    
    
def create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,experiment_name="unfiled",alphas=[0.1],seed=-1,max_examples=100):
    """Creates a set of TCAV vectors based on concepts, with the intent of predicting target
    
    Arguments:
        concepts: List of Concept Strings; each represents an ImageNet class, e.g. ["zebra", ...]
        target: One ImageNet Class String; what we intend to predict using these concepts
        model_name: Mode used; currently only GoogleNet
        bottlenecks: Which layers in the GoogleNet model to use; such as ["mixed4c"]
        num_random_exp: Number of comparisons between each concept and the random classes; 
            The number of concept vectors made is roughly the num_random_exp
        max_examples: How many images are in each positive image class
            
    Returns:
        None
        
    Side Effects:
        Trains a set of concept_vectors stored at ./results/cavs
    """
    
    if seed == -1:
        seed = random.randint(0,100000)
    
    np.random.seed(seed)
    random.seed(seed)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:

        act_generator = load_activations_model(experiment_name,max_examples,model_name,sess)
        delete_previous_activations(bottlenecks[0],concepts)

        cav_dir = './results/cavs/{}/{}'.format(experiment_name,seed)
        if not os.path.exists(cav_dir):
            os.makedirs(cav_dir)
        else:
            reset_tcav_vectors(concepts,num_random_exp,experiment_name,seed,bottlenecks[0],alphas[0])

        mytcav = tcav.TCAV(sess,
                       target,
                       concepts,
                       bottlenecks,
                       act_generator,
                       alphas,
                       cav_dir=cav_dir,
                       num_random_exp=num_random_exp)#10)

        # Reset the runs so it doesn't compare random-random
        mytcav.relative_tcav = True
        mytcav._process_what_to_run_expand(num_random_exp=num_random_exp+1)
        mytcav.params = mytcav.get_params()

        mytcav.run(run_parallel=False)
        
def create_model_vectors(attributes,dataset,suffix,seed=-1):
    """Develop concept vectors based on a model representation
    
    Arguments:
        attributes: List of attributes to create concept vectors for
        dataset: Object from the dataset class
        suffix: String representing which specific instance of the dataset we're testing
        seed: Random number seed
        
    Returns: Nothing
    
    Side Effects: Saves concpet vectors to results/model_vectors/seed/attribute.npy"""
    
    
    max_images = 25
    model = "VGG16"
    
    if suffix == "_model_robustness":
        model = "VGG16Robustness"
    elif suffix == "_model_responsiveness":
        model = "VGG16Responsiveness"
        
    for attribute in attributes:
        create_folder_from_attribute(attribute,
                                     dataset.get_images_with_attribute,num_images=max_images,suffix=suffix,seed=seed)
    
    with tf.compat.v1.Session() as sess:
        activation_generator = load_activations_model(dataset.experiment_name+suffix,max_images,model,sess)
        activations = get_activations_dictionary(attributes,
                                                 sess,
                                                 model_name=model,
                                                 experiment_name=dataset.experiment_name,
                                                 max_examples=max_images)
    
    if not os.path.exists("results/model_vectors/{}/{}".format(dataset.experiment_name+suffix,seed)):
        os.makedirs("results/model_vectors/{}/{}".format(dataset.experiment_name+suffix,seed))
        
    for attribute in activations:
        save_file = "results/model_vectors/{}/{}/{}.npy".format(dataset.experiment_name+suffix,seed,attribute)
        np.save(open(save_file,"wb"),activations[attribute])

def create_dimensionality_reduced_TCAV(dataset,suffix,seed,k=8):
    """Create a version of the TCAV vectors that have PCA dimensionality reduce them to k dimensions
    
    Arguments:
        dataset: An object from the dataset class, for which to create vectors
        suffix: String for a specific version of the dataset
        seed: Number that represents a random seed
        k: Size of the dimensionality reduced vectors
        
    Returns: Nothing
    
    Side Effects: Stores the vectors at tcav_dim_reduce folder
    """
    
    folder = "results/tcav_dim_reduce/{}/{}".format(dataset.experiment_name+suffix,seed)
    attributes = dataset.get_attributes()
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    index_by_attribute = [0]

    all_vectors = None

    for attribute in attributes:
        vectors = load_tcav_vectors_simple(attribute,dataset,suffix,seed=seed)
        index_by_attribute.append(len(vectors) + index_by_attribute[-1])

        if type(all_vectors) == type(None):
            all_vectors = vectors
        else:
            all_vectors = np.concatenate([all_vectors,vectors])
    X_embedded = PCA(n_components=k).fit_transform(all_vectors)
    for i in range(len(index_by_attribute)-1):
        start = index_by_attribute[i]
        end = index_by_attribute[i+1]
        attribute = attributes[i]
        vectors = X_embedded[start:end,:]

        file_name = folder + "/{}.npy".format(attribute)

        np.save(open(file_name,"wb"),vectors)

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
        return np.load(open("results/{}/{}/{}/{}.npy".format(folder_name,dataset.experiment_name+suffix,seed,attribute),"rb"),allow_pickle=True)

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
    
def load_tcav_dr_vectors_simple(attribute,dataset,suffix,seed=-1):
    """Develop a concept vectors that's a dimensionality reduced version of TCAV
    
    Arguments:
        attribute: A single attribute 
        dataset: Object from the dataset class
        suffix: String; which specific instance of the dataset are we testing out 
    
    Returns: 
        Numpy array of dimensionality reduced TCAV vectors
    """
        
    return load_vector_from_folder("tcav_dim_reduce")(attribute,dataset,suffix,seed=seed)

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
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--class_name', type=str, default='',
                        help='Name of the ImageNet class for which concept vectors are generated.')
    parser.add_argument('--target', type=str,
                        help='Target which the concept vectors are aiming to predict.',default='zebra')
    parser.add_argument('--model_name', type=str,
                        help='Name of the model used to generate concept vectors',default="VGG16")
    parser.add_argument('--alpha',type=float,
                        help='Regularization parameter to train concept vector',default=0.1)
    parser.add_argument('--bottleneck',type=str,
                        help='Layer of model used when training concept vectors; activations are taken from this',
                        default='block4_conv1')
    parser.add_argument('--num_random_exp', type=int,
                        help='Number of random ImageNet classes we compare the concept vector with')
    parser.add_argument('--images_per_folder',type=int, default=50,
                        help='Number of images in each random random folder')
    parser.add_argument('--seed',type=int, default=42,
                        help='Random seed used in tcav experiment')

    args = parser.parse_args()
            
    if args.algorithm == 'tcav':
        create_tcav_vectors([args.class_name],args.target,args.model_name[args.bottleneck],
                            args.num_random_exp,alphas=[args.alpha],seed=args.seed)
    elif 'tcav_cub' in args.algorithm:
        suffix = args.algorithm.replace("tcav_cub","")
        
        create_tcav_dataset(args.class_name,CUB_Dataset(),
                            args.num_random_exp,args.images_per_folder,
                            seed=args.seed,suffix=suffix,model_name=args.model_name,bottlenecks=[args.bottleneck])
    elif 'tcav_mnist' in args.algorithm:
        suffix = args.algorithm.replace("tcav_mnist","")
        create_tcav_dataset(args.class_name,MNIST_Dataset(),
                            args.num_random_exp,args.images_per_folder,
                            seed=args.seed,suffix=suffix,model_name=args.model_name,bottlenecks=[args.bottleneck])
    elif 'model_mnist' in args.algorithm:
        suffix = args.algorithm.replace("model_mnist","")
        dataset = MNIST_Dataset()
        attributes = dataset.get_attributes()
        create_model_vectors(attributes,dataset,suffix,args.seed)
    elif 'model_cub' in args.algorithm:
        suffix = args.algorithm.replace("model_cub","")
        dataset = CUB_Dataset()
        attributes = dataset.get_attributes()
        create_model_vectors(attributes[:len(attributes)//2],dataset,suffix,args.seed)
        create_model_vectors(attributes[len(attributes)//2:],dataset,suffix,args.seed)
    else:
        raise Exception("{} not implemented to generate concept vectors".format(args.algorithm))
