from src.hierarchy import *
from src.concept_vectors import *
from src.dataset import *
import numpy as np
import itertools
from scipy import stats
import tcav.utils as utils
from scipy.spatial.distance import cdist
import scipy
import time

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
        Float, representing the average pairwise distance between hierarchies, and the standard deviation
    """
    
    all_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,"",attributes,seed) for seed in random_seeds]
    
    distance_list = []
    for h1,h2 in itertools.combinations(all_hierarchies,r=2):
        distance = h1.distance(h2)
        distance_list.append(distance)
        
    return np.mean(distance_list), np.std(distance_list)

def compare_same_images_by_suffix(hierarchy_method,embedding_method,dataset,attributes,random_seeds,suffix,baseline_hierarchies=None):
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
        Float, representing average paired distance between hierarchies, and the standard deviation
    """
    
    if suffix not in ["","_image_robustness","_image_responsiveness","_model_robustness","_model_responsiveness"]:
        raise Exception("{} suffix not supported".format(suffix))
        
    if baseline_hierarchies == None:
        baseline_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,"",attributes,seed) for seed in random_seeds]
    robust_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,suffix,attributes,seed) for seed in random_seeds]
        
    distance_list = []
    for h1,h2 in zip(baseline_hierarchies,robust_hierarchies):
        distance_list.append(h1.distance(h2))
                
    return np.mean(distance_list), np.std(distance_list)
    
def robustness_image_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
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
                                         embedding_method,dataset,attributes,random_seeds,"_image_robustness",
                                        baseline_hierarchies=baseline_hierarchies)
    
def responsiveness_image_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
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
                                         embedding_method,dataset,attributes,random_seeds,"_image_responsiveness",
                                        baseline_hierarchies=baseline_hierarchies)

def robustness_model_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
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
                                         embedding_method,dataset,attributes,random_seeds,"_model_robustness",
                                        baseline_hierarchies=baseline_hierarchies)


def responsiveness_model_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
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
                                         embedding_method,dataset,attributes,random_seeds,"_model_responsiveness",
                                        baseline_hierarchies=baseline_hierarchies)


def reset_dataset(dataset,seed,max_images):
    """Reset the images and the activations for a dataset, then redownload them
    
    Arguments:
        dataset: Object from the Dataset class
        seed: Random seed that determines which images are selected for the dataset
        max_images: How many images we'll have, at most, in a folder

    Returns: Nothing
    
    Side Effects: Deletes all activations and re-downloads all images
    """
    
    activation_dir = './results/activations'
    
    concepts = dataset.get_attributes()
    for attribute in concepts:
        create_folder_from_attribute(attribute,dataset.get_images_with_attribute,seed=seed,
                                        suffix='',num_images=max_images)
        
    
    bottlenecks = ['block4_conv1']
    # Remove existing activations
    for concept in concepts:
        activation_file_location = "{}/acts_{}_{}".format(activation_dir,concept,bottlenecks[0])
        if os.path.exists(activation_file_location):
            os.remove(activation_file_location)

def find_similar_concepts(concept,dataset,activations,num_similar_concepts,seed,metric='cosine'):
    """Find the num_similar_concepts most similar concepts, ranked by their similarity in 
    some layer of the Neural Network 
    
    Arguments:
        concept: Baseline concept for which we're trying to find the K most similar concepts
        dataset: String, such as 'mnist'
        activations: Dictionary of activations for each concept in some layer 
        similar_concepts: How many similar concepts we should find 

    Returns:
        List, ordering the concepts from closest to furthest
    """
    
    activation_dir = './results/activations'
    working_dir = './results/tmp'
    image_dir = "./dataset/images"
        
    concepts = dataset.get_attributes()
                
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
        dataset: Object from the dataset class
        concept: Fixed concept which we're comparing other concept to
        co_occuring_concepts: List of concepts which we want to rank
        seed: Random seed to create embeddings
    
    Returns:
        List, which ranks the elements of co_occuring_concepts from smallest to largest
    """
    
    embedding_constant = embedding_method(concept,dataset,"",seed=seed)
    embedding_by_concept = {}
    
    for other_concept in co_occuring_concepts:
        embedding_by_concept[other_concept] = embedding_method(other_concept,dataset,"",seed=seed)
        
    metric_concept_pairs = [(other_concept,
                             np.mean(cdist(embedding_constant,embedding_by_concept[other_concept],metric=metric)))
                             for other_concept in co_occuring_concepts]

    metric_concept_pairs = sorted(metric_concept_pairs, key=lambda k: k[1])
    return [i[0] for i in metric_concept_pairs]

def truthfulness_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds,model="VGG16",baseline_hierarchies=None):
    """Compute the truthfulness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        hierarchy_method: Function such as create_ward_hierarchy that creates a dendrogram
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: Object from the dataset class
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing similarity between distances in the model, and the distances predicted by the hierarchy; it's an average correlation between 0-1, and the standard deviation
    """
        
    n_concepts = 5
    compare_concepts = 3
    max_images = 25
    
    
    avg_truthfulness = []
    
    reset_dataset(dataset,random_seeds[0],100)

    with tf.compat.v1.Session() as sess:
        activations = get_activations_dictionary(dataset.get_attributes(),
                                                 sess,
                                                 model_name=model,
                                                 experiment_name=dataset.experiment_name,
                                                 max_examples=max_images)
            
    for seed in random_seeds:  
        random.seed(seed)
        np.random.seed(seed)

        selected_concepts = random.sample(attributes,k=n_concepts)
        
        temp_truthfulness = []
        
        for concept in selected_concepts:
            co_occuring_concepts = find_similar_concepts(concept,dataset,activations,compare_concepts,seed)                        
            co_occuring_concepts_hierarchy = rank_distance_concepts(embedding_method,dataset,concept,
                                                                    co_occuring_concepts,seed)
            
            temp_truthfulness.append(stats.kendalltau(co_occuring_concepts,
                                                     co_occuring_concepts_hierarchy).correlation)
            
        avg_truthfulness.append(np.mean(temp_truthfulness))
            
        
    return np.mean(avg_truthfulness), np.std(avg_truthfulness)

def compute_all_metrics(hierarchy_method,embedding_method,dataset,attributes,random_seeds,model="VGG16"):
    metrics = [stability_metric,robustness_image_metric,
               responsiveness_image_metric,robustness_model_metric,responsiveness_model_metric,
              truthfulness_metric]
    metric_names = ['Stability', 'Image Robustness', 
                'Image Responsiveness','Model Robustness','Model Responsiveness',
                   'Truthfulness']
        
    baseline_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,"",attributes,seed) for seed in random_seeds]
    
    results = {}
    
    for metric,name in zip(metrics,metric_names):
        score = metric(hierarchy_creation_method,
                        load_label_vectors_simple,
                        dataset,
                        attributes,
                        seeds, 
                        baseline_hierarchies=baseline_hierarchies)
        print("{}: {}".format(name, score))
        results[name] = score
        
    return results