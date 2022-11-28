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

def load_cem_vectors(concept_number):
    """Load all the 'active' embeddings from Concept Embedding Models
    
    Arguments:
        concept_number: An index for which concept is used
            Should correspond to the index in the 'c' array in CEM

    Returns:
        Numpy array of size (k,n), where k = number of concept vectors, and n = size of embedding
    """
    dataset_location = "../sandbox/cem/concepts"
    
    file_location = dataset_location + "/concept_{}_active.npy".format(concept_number)
    return np.load(open(file_location,"rb"))

def load_tcav_vectors(concept,bottlenecks,alphas=[0.1]):
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
    dataset_location = "./results/cavs"
    all_matching_files = [] 
    concept_meta_info = []
    
    for bottleneck in bottlenecks:
        for alpha in alphas:
            file_name_pattern = "{}-random*-{}-linear-{}.pkl".format(concept,bottleneck,alpha)
            all_matching_files += glob.glob(dataset_location+"/"+file_name_pattern)
                        
            for file_name in all_matching_files:
                re_search = re.search('{}-random(.*)-{}-linear-{}.pkl'.format(concept,bottleneck,alpha),file_name)
                random_concept = re_search.group(1)
                concept_meta_info.append({'alpha': alpha,'bottleneck': bottleneck, 'random_concept': random_concept, 'concept': concept})

    all_concept_vectors = []
    
    for file in all_matching_files:
        concept_vector = pickle.load(open(file,"rb"))['cavs'][0]
        all_concept_vectors.append(concept_vector)

    all_concept_vectors = np.array(all_concept_vectors)
    return all_concept_vectors, concept_meta_info
    
def create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,alphas=[0.1]):
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
    
    cav_dir = './results/cavs'
    activation_dir = './results/activations'
    working_dir = './results/tmp'
    image_dir = "./dataset/images"
    
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
    
    mytcav = tcav.TCAV(sess,
                   target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)#10)
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

    args = parser.parse_args()
    
    if args.algorithm not in ['tcav']:
        raise Exception("{} not implemented to generate concept vectors".format(parser.algorithm))
    
    if args.algorithm == 'tcav':
        create_tcav_vectors([args.class_name],args.target,args.model_name,[args.bottleneck],args.num_random_exp,alphas=[args.alpha])

   