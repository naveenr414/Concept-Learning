from src.hierarchy import *
from src.concept_vectors import *
from src.dataset import *
import numpy as np
import itertools
from scipy import stats
import tcav.utils as utils
from scipy.spatial.distance import cdist
import scipy

def stability_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the stability metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments:
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings
    
    Returns:
        Float, representing the average pairwise distance between hierarchies
    """
    
    all_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,attributes,seed) for seed in random_seeds]
    
    distance_list = []
    for h1,h2 in itertools.combinations(all_hierarchies,r=2):
        distance = h1.distance(h2)
        distance_list.append(distance)
        
    return np.mean(distance_list)

def compare_same_images_by_suffix(hierarchy_method,embedding_method,dataset,attributes,random_seeds,suffix):
    """Compare hierarchies by the same seed, with different dataset suffixes
        Use cases include comparing mnist vs. mnist_image_robustness, using pairwise 
        
    Arguments:
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings
        suffix: Qualifier for which dataset subtype we're looking at, such as _image_robustness
        
    Returns: 
        Float, representing average paired distance between hierarchies
    """
    
    if suffix not in ["","_image_robustness","_image_responsiveness","_model_robustness","_model_responsiveness"]:
        raise Exception("{} suffix not supported".format(suffix))
    
    baseline_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset.experiment_name,attributes,seed) for seed in random_seeds]
    robust_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset.experiment_name+suffix,attributes,seed) for seed in random_seeds]
    
    distance_list = []
    for h1,h2 in zip(baseline_hierarchies,robust_hierarchies):
        distance_list.append(h1.distance(h2))
        
    return np.mean(distance_list)
    
def robustness_image_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the image robustness for  metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing the average distance between each pair of images when perturbed, comparing only like-seeds
    """
    
    return compare_same_images_by_suffix(hierarchy_method,
                                         embedding_method,dataset,attributes,random_seeds,"_image_robustness")
    
def responsiveness_image_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the image responsiveness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
         Float, representing the average distance between each pair of images when significantly altered, comparing only like-seeds
    """

    return compare_same_images_by_suffix(hierarchy_method,
                                         embedding_method,dataset,attributes,random_seeds,"_image_responsiveness")

def robustness_model_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the model robustness for metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing the average distance between each pair of images when perturbed, comparing only like-seeds
    """
    
    return compare_same_images_by_suffix(hierarchy_method,
                                         embedding_method,dataset,attributes,random_seeds,"_model_robustness")


def responsiveness_model_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the model responsiveness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
         Float, representing the average distance between each pair of images when significantly altered, comparing only like-seeds
    """

    return compare_same_images_by_suffix(hierarchy_method,
                                         embedding_method,dataset,attributes,random_seeds,"_model_responsiveness")


def reset_dataset(dataset,seed,max_images):
    """Reset the images and the activations for a dataset, then redownload them
    
    Arguments:
        dataset: String representing a dataset, such as 'mnist'
        seed: Random seed that determines which images are selected for the dataset
        max_images: How many images we'll have, at most, in a folder

    Returns: Nothing
    
    Side Effects: Deletes all activations and re-downloads all images
    """
    
    activation_dir = './results/activations'
    
    if dataset == 'mnist':
        concepts = get_mnist_attributes()
        for attribute in concepts:
            create_folder_from_attribute(attribute,get_mnist_images_by_attribute,seed=seed,
                                            suffix='',num_images=max_images)
    else:
        raise Exception("Dataset {} not implemented yet".format(dataset))
        
    
    bottlenecks = ['mixed4c']
    # Remove existing activations
    for concept in concepts:
        activation_file_location = "{}/acts_{}_{}".format(activation_dir,concept,bottlenecks[0])
        if os.path.exists(activation_file_location):
            os.remove(activation_file_location)

def find_similar_concepts(concept,dataset,model,num_similar_concepts,seed,metric='cosine'):
    """Find the num_similar_concepts most similar concepts, ranked by their similarity in 
    some layer of the Neural Network 
    
    Arguments:
        concept: Baseline concept for which we're trying to find the K most similar concepts
        dataset: String, such as 'mnist'
        model: String, representing the model name, such as 'GoogleNet' 
        similar_concepts: How many similar concepts we should find 

    Returns:
        List, ordering the concepts from closest to furthest
    """
    
    activation_dir = './results/activations'
    working_dir = './results/tmp'
    image_dir = "./dataset/images"
    
    max_images = 100
    
    if dataset == 'mnist':
        concepts = get_mnist_attributes()
    
    activations = load_activations_tcav(concepts,experiment_name=dataset,max_examples=max_images)
    our_vectors = activations[concept]
    
    all_distance_pairs = []
    for other_concept in concepts:
        if other_concept != concept:
            distances = cdist(our_vectors,activations[other_concept],metric=metric)
            all_distance_pairs.append((other_concept,np.mean(distances)))
            
    all_distance_pairs = sorted(all_distance_pairs,key=lambda k: k[1])
    return [i[0] for i in all_distance_pairs][:num_similar_concepts]

def rank_distance_concepts(embedding_method,dataset,concept,co_occuring_concepts,seed,metric='cosine'):
    """Order how close certain concepts are to a fixed concept, given an embedding method + seed
    
    Arguments:
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        concept: Fixed concept which we're comparing other concept to
        co_occuring_concepts: List of concepts which we want to rank
        seed: Random seed to create embeddings
    
    Returns:
        List, which ranks the elements of co_occuring_concepts from smallest to largest
    """
    
    embedding_constant = embedding_method(concept,dataset,seed)
    embedding_by_concept = {}
    
    for other_concept in co_occuring_concepts:
        embedding_by_concept[other_concept] = embedding_method(other_concept,dataset,seed)
        
    metric_concept_pairs = [(other_concept,
                             np.mean(cdist(embedding_constant,embedding_by_concept[other_concept],metric=metric)))
                             for other_concept in co_occuring_concepts]

    metric_concept_pairs = sorted(metric_concept_pairs, key=lambda k: k[1])
    return [i[0] for i in metric_concept_pairs]

def truthfulness_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,model="GoogleNet"):
    """Compute the truthfulness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing similarity between distances in the model, and the distances predicted by the hierarchy; it's an average correlation between 0-1 
    """
    
    random.seed(42)
    
    n_concepts = len(attributes)//5
    compare_concepts = len(attributes)//5
    
    selected_concepts = random.sample(attributes,k=n_concepts)
    
    avg_truthfulness = []
    
    for seed in random_seeds:
        #reset_dataset(dataset,seed,100)
        for concept in selected_concepts:
            co_occuring_concepts = find_similar_concepts(concept,dataset,model,compare_concepts,seed)
            co_occuring_concepts_hierarchy = rank_distance_concepts(embedding_method,dataset,concept,
                                                                    co_occuring_concepts,seed)
            
            avg_truthfulness.append(stats.kendalltau(co_occuring_concepts,
                                                     co_occuring_concepts_hierarchy).correlation)
        
    return np.mean(avg_truthfulness)
        