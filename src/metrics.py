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