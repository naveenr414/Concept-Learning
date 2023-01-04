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
    dataset_location = "./results/cavs/{}/{}".format(experiment_name,seed)
    all_matching_files = [] 
    concept_meta_info = []
    
    for bottleneck in bottlenecks:
        for alpha in alphas:
            file_name_pattern = "{}-random*-{}-linear-{}.pkl".format(concept,bottleneck,alpha)
            all_matching_files += glob.glob(dataset_location+"/"+file_name_pattern)
                        
            for file_name in all_matching_files:
                re_search = re.search('{}-random(.*)-{}-linear-{}.pkl'.format(concept,bottleneck,alpha),file_name)
                random_concept = re_search.group(1)
                concept_meta_info.append({'alpha': alpha,'bottleneck': bottleneck, 'random_concept': int(random_concept), 'concept': concept})

    concept_meta_info, all_matching_files = zip(*sorted(zip(concept_meta_info,all_matching_files), key=lambda k: k[0]['random_concept']))
                
    all_concept_vectors = []
    
    for file in all_matching_files:
        concept_vector = pickle.load(open(file,"rb"))['cavs'][0]
        all_concept_vectors.append(concept_vector)

    all_concept_vectors = np.array(all_concept_vectors)
    return all_concept_vectors, concept_meta_info

def create_vector_from_label_cub(attribute_name,seed=-1):
    """Generate sparse concept vectors, by looking at whether a concept is present in a data point
        This produces a 0-1 vector, with the vector <0,1,0> representing
        the presence of the attribute in data point 1, and not present in datapoints 0, 2
        
    Arguments:
        attribute_name: String representing one of the 112 CUB attributes

    Returns:
        concept_vector: Numpy vector representing the concept vector for the attribute
    """
    
    all_attributes = get_cub_attributes()
    if attribute_name not in all_attributes:
        raise Exception("Unable to generate vector from attribute {}".format(attribute_name))
    index = all_attributes.index(attribute_name)
    
    train_data = load_cub_split('train',seed=seed)
    concept_vector = [i['attribute_label'][index] for i in train_data]
    return np.array(concept_vector).reshape((1,len(concept_vector)))

def create_vector_from_label_mnist(attribute_name,seed=-1):
    """Generate sparse concept vectors, by looking at whether a concept is present in a data point
        This produces a 0-1 vector, with the vector <0,1,0> representing
        the presence of the attribute in data point 1, and not present in datapoints 0, 2
        
    Arguments:
        attribute_name: String representing one of the 112 CUB attributes

    Returns:
        concept_vector: Numpy vector representing the concept vector for the attribute
    """
    
    mnist_data = load_mnist(seed)    
    concept_vector = [i[attribute_name] for i in mnist_data]
    return np.array(concept_vector).reshape((1,len(concept_vector)))

def create_tcav_cub(attribute_name,num_random_exp,images_per_folder=50,seed=-1):
    """Helper function to create TCAV from CUB Attribute
        It creates the folder with images for the attribute, trains the TCAV vector,
        then deletes the folder
    
    Arguments:
        attribute_name: String containing one of the 112 CUB attributes

    Returns:
        None
        
    Side Effects:
        Trains a set of concept vectors, stored at ./results/cavs/experiment_name/seed
    """
    
    create_folder_from_attribute(attribute_name,get_cub_images_by_attribute,seed)
    create_random_folder_without_attribute(attribute_name,num_random_exp,get_cub_images_without_attribute,images_per_folder,seed=seed)
    
    concepts = [attribute_name]
    target = "zebra"
    model_name = "GoogleNet"
    bottlenecks = ["mixed4c"]
    alphas = [0.1]
    
    create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,experiment_name="cub",alphas=[0.1],seed=seed)

def create_tcav_mnist(attribute_name,num_random_exp,images_per_folder=50,seed=-1):
    """Helper function to create TCAV from CUB Attribute
        It creates the folder with images for the attribute, trains the TCAV vector,
        then deletes the folder
    
    Arguments:
        attribute_name: String containing one of the 112 CUB attributes

    Returns:
        None
        
    Side Effects:
        Trains a set of concept vectors, stored at ./results/cavs/experiment_name/seed
    """
    
    create_folder_from_attribute(attribute_name,get_mnist_images_by_attribute,seed=seed)
    create_random_folder_without_attribute(attribute_name,num_random_exp,get_mnist_images_without_attribute_one_class,images_per_folder,seed=seed)
    
    concepts = [attribute_name]
    target = "zebra"
    model_name = "GoogleNet"
    bottlenecks = ["mixed4c"]
    alphas = [0.1]
    
    create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,experiment_name="mnist",alphas=[0.1],seed=seed)

def load_activations_tcav(attribute_list,experiment_name="unfiled",seed=-1):
    """From a list of concepts or attributes, generate their representation in some model
        such as GoogleNet, at some bottleneck layer
        
    Arguments:
        attribute_list: String list of concepts to be analyzed, such as 'zebra'

    Returns:
        Dictionary: Each key is an attribute, and each value is a numpy array of size 
            n x 100K, where n is the number of concept vectors, and 100K is the size of the 
            bottleneck
    """
    
    cav_dir = './results/cavs/{}/{}'.format(experiment_name,seed)
    activation_dir = './results/activations'
    working_dir = './results/tmp'
    image_dir = "./dataset/images"
    model_name = "GoogleNet"
    
    sess = utils.create_session()
    if model_name == "GoogleNet":
        GRAPH_PATH = "./dataset/models/inception5h/tensorflow_inception_graph.pb"
        LABEL_PATH = "./dataset/models/inception5h/imagenet_comp_graph_label_strings.txt"
        mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)
    else:
        raise Exception("create_tcav_vectors not implemented for {}".format(model_name))

    act_generator = act_gen.ImageActivationGenerator(mymodel, image_dir, activation_dir,max_examples=500)
    
    bottlenecks = ["mixed4c"]
    
    acts = {}
    for i in attribute_list:
        examples = act_generator.get_examples_for_concept(i)
        activation_examples = act_generator.model.run_examples(examples, bottlenecks[0])
        acts[i] = activation_examples
        shape = acts[i].shape
        acts[i] = acts[i].reshape((shape[0],shape[1]*shape[2]*shape[3]))
        print("Activation {} is shape {}".format(i,acts[i].shape))
    
    return acts
    
def create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,experiment_name="unfiled",alphas=[0.1],seed=-1):
    """Creates a set of TCAV vectors based on concepts, with the intent of predicting target
    
    Arguments:
        concepts: List of Concept Strings; each represents an ImageNet class, e.g. ["zebra", ...]
        target: One ImageNet Class String; what we intend to predict using these concepts
        model_name: Mode used; currently only GoogleNet
        bottlenecks: Which layers in the GoogleNet model to use; such as ["mixed4c"]
        num_random_exp: Number of comparisons between each concept and the random classes; 
            The number of concept vectors made is roughly the num_random_exp
            
    Returns:
        None
        
    Side Effects:
        Trains a set of concept_vectors stored at ./results/cavs
    """
    
    if seed == -1:
        seed = random.randint(0,100000)
    
    np.random.seed(seed)
    random.seed(seed)
    
    cav_dir = './results/cavs/{}/{}'.format(experiment_name,seed)
    activation_dir = './results/activations'
    working_dir = './results/tmp'
    image_dir = "./dataset/images"
    
    if not os.path.exists(cav_dir):
        os.makedirs(cav_dir)
    
    sess = utils.create_session()
    if model_name == "GoogleNet":
        GRAPH_PATH = "./dataset/models/inception5h/tensorflow_inception_graph.pb"
        LABEL_PATH = "./dataset/models/inception5h/imagenet_comp_graph_label_strings.txt"
        mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)
    else:
        raise Exception("create_tcav_vectors not implemented for {}".format(model_name))

    act_generator = act_gen.ImageActivationGenerator(mymodel, image_dir, activation_dir, max_examples=100)
    
    # Remove existing TCAV vectors, so it generates new ones
    for concept in concepts:
        for i in range(num_random_exp):
            tcav_file_location = "{}/{}-random500_{}-{}-linear-{}.pkl".format(cav_dir,concept,i,bottlenecks[0],alphas[0])
            if os.path.exists(tcav_file_location):
                os.remove(tcav_file_location)
    
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
    
    print("Pairs to test {}".format(mytcav.pairs_to_test))
    print("Params {}".format(mytcav.params))

    mytcav.run(run_parallel=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--class_name', type=str,
                        help='Name of the ImageNet class for which concept vectors are generated.')
    parser.add_argument('--target', type=str,
                        help='Target which the concept vectors are aiming to predict.',default='zebra')
    parser.add_argument('--model_name', type=str,
                        help='Name of the model used to generate concept vectors',default="GoogleNet")
    parser.add_argument('--alpha',type=float,
                        help='Regularization parameter to train concept vector',default=0.1)
    parser.add_argument('--bottleneck',type=str,
                        help='Layer of model used when training concept vectors; activations are taken from this',
                        default='mixed4c')
    parser.add_argument('--num_random_exp', type=int,
                        help='Number of random ImageNet classes we compare the concept vector with')
    parser.add_argument('--images_per_folder',type=int, default=50,
                        help='Number of images in each random random folder')

    args = parser.parse_args()
    
    if args.algorithm not in ['tcav','tcav_cub','tcav_mnist']:
        raise Exception("{} not implemented to generate concept vectors".format(args.algorithm))
        
    if args.algorithm == 'tcav':
        create_tcav_vectors([args.class_name],args.target,args.model_name,[args.bottleneck],args.num_random_exp,alphas=[args.alpha])
    elif args.algorithm == 'tcav_cub':
        create_tcav_cub(args.class_name,args.num_random_exp,args.images_per_folder)
    elif args.algorithm == 'tcav_mnist':
        create_tcav_mnist(args.class_name,args.num_random_exp,args.images_per_folder)
   