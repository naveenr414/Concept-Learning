import tcav
import numpy as np
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
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

class ResnetWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])
    
class VGGWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])

def load_cem_vectors(experiment_name,concept_number,seed=-1):
    """Load all the 'active' embeddings from Concept Embedding Models
    
    Arguments:
        experiment_name: Which experiment these concepts come from
            Concept files are of form {experiment_name}_concept...
        concept_number: An index for which concept is used
            Should correspond to the index in the 'c' array in CEM

    Returns:
        Numpy array of size (k,n), where k = number of concept vectors, and n = size of embedding
    """
    dataset_location = "results/cem_concepts"
    
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

def create_vector_from_label(attribute_name,dataset,seed=-1):
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
    
    train_data = dataset.get_data(seed=seed)
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
            GRAPH_PATH = "./dataset/models/resnet50/baseline.h5"
        elif model_name == "Resnet50Robustness":
            GRAPH_PATH = "./dataset/models/resnet50/robust.h5"
        elif model_name == "Resnet50Responsiveness":
            GRAPH_PATH = "./dataset/models/resnet50/responsive.h5"
            
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
                               experiment_name="unfiled",max_examples=500,bottleneck="block4_conv1"):
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
    
    delete_previous_activations(bottleneck,attribute_list)
    act_generator = load_activations_model(experiment_name,max_examples,model_name,sess)
            
    acts = {}
    for i in attribute_list:
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

def load_tcav_vectors_simple(attribute,dataset,seed=-1):
    """Simplified call to load_tcav_vectors that is standardized across embeddings
    
    Arguments:
        attribute: Which TCAV concept we're looking to get vectors for, as a string
        dataset: Object from the datastet class
    
    Returns: 
        Numpy array of TCAV vectors
    """
        
    return load_tcav_vectors(attribute,['block4_conv1'],experiment_name=dataset.experiment_name,seed=seed)[0]

def load_label_vectors_simple(attribute,dataset,seed=-1):
    """Simplified call to create_vector_from_label_cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    vector = create_vector_from_label(attribute,dataset,seed=seed)
    
    return np.array(vector)
            
def load_cem_vectors_simple(attribute,dataset,seed=-1):
    """Simplified call to create vector from cub/mnist that is standardized across embeddings
    
    Arguments:
        attribute: Which concept we're looking to get vectors for, as a string
        dataset: Object from the dataset class
    
    Returns: 
        Numpy array of label-based vectors
    """
    
    all_attributes = dataset.get_attributes()
    attribute_index = all_attributes.index(attribute)
    return load_cem_vectors(dataset.experiment_name,attribute_index,seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--class_name', type=str,
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
    elif args.algorithm == 'tcav_cub':
        suffix = args.algorithm.replace("tcav_cub","")
        create_tcav_dataset(args.class_name,CUB_Dataset(),
                            args.num_random_exp,args.images_per_folder,
                            seed=args.seed,suffix=suffix,model_name=args.model_name,bottlenecks=[args.bottleneck])
    elif 'tcav_mnist' in args.algorithm:
        suffix = args.algorithm.replace("tcav_mnist","")
        create_tcav_dataset(args.class_name,MNIST_Dataset(),
                            args.num_random_exp,args.images_per_folder,
                            seed=args.seed,suffix=suffix,model_name=args.model_name,bottlenecks=[args.bottleneck])
    else:
        raise Exception("{} not implemented to generate concept vectors".format(args.algorithm))

