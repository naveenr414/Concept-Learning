from src.hierarchy import *
from src.concept_vectors import *
import numpy as np
import itertools

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
    
    if suffix not in ["","_image_robustness","_image_responsiveness"]:
        raise Exception("{} suffix not supported".format(suffix))
    
    baseline_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset,attributes,seed) for seed in random_seeds]
    robust_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset+suffix,attributes,seed) for seed in random_seeds]
    
    responsiveness_hierarchies = [create_hierarchy(hierarchy_method,embedding_method,dataset+"_image_responsiveness",attributes,seed) for seed in random_seeds]

    print(baseline_hierarchies[0].distance(responsiveness_hierarchies[0]))
    print(baseline_hierarchies[0].distance(robust_hierarchies[0]))

    print(robust_hierarchies[0].distance(baseline_hierarchies[0]))
    print(responsiveness_hierarchies[0].distance(baseline_hierarchies[0]))

    
    distance_list = []
    for h1,h2 in zip(baseline_hierarchies,robust_hierarchies):
        distance_list.append(h1.distance(h2))
        
    return np.mean(distance_list)
    
def robustness_image_metric(hierarchy_method,embedding_method,dataset,attributes,random_seeds):
    """Compute the robustness for  metric for a set of random seeds, given a hierarchy+embedding method
    
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
    
    return compare_same_images_by_suffix(hierarchy_method,embedding_method,dataset,attributes,random_seeds,"_image_robustness")
    
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

    return compare_same_images_by_suffix(hierarchy_method,embedding_method,dataset,attributes,random_seeds,"_image_responsiveness")

        