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

def get_top_k_pairs(embedding,k=3):
    pairs = []
    
    for i,row in enumerate(embedding):
        top_k = np.argpartition(row, k+1)[:k+1]
        top_k = [j for j in top_k if j != i]
        
        assert len(top_k) == k
        
        pairs += [(i,j) for j in top_k]

    return pairs 

def embedding_distance(h1,h2,k=3):
    pairs_1 = get_top_k_pairs(h1,k=k)
    pairs_2 = get_top_k_pairs(h2,k=k)
    
    intersection = set(pairs_1).intersection(set(pairs_2))
    
    return 1 - len(intersection)/(k*len(h1))
    

def stability_metric(embedding_method,dataset,attributes,random_seeds):
    """Compute the stability metric for a set of random seeds, given a hierarchy+embedding method

    Arguments:
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing the average pairwise distance between hierarchies, and the standard deviation
    """

    baseline_embeddings = [flat_distance_to_square(get_concept_distances
                                                   (embedding_method,dataset,"",attributes,seed)) for seed in random_seeds]

    distance_list = []
    for h1,h2 in itertools.combinations(baseline_embeddings,r=2):
        distance = embedding_distance(h1,h2)
        distance_list.append(distance)
        
    return np.mean(distance_list), np.std(distance_list)

def compare_same_images_by_suffix(embedding_method,dataset,attributes,random_seeds,suffix,baseline_hierarchies=None):
    """Compare hierarchies by the same seed, with different dataset suffixes
        Use cases include comparing mnist vs. mnist_image_robustness, using pairwise 
        
    Arguments:
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
        
    baseline_embeddings = [flat_distance_to_square(get_concept_distances
                                                   (embedding_method,dataset,"",attributes,seed)) for seed in random_seeds]
    robust_hierarchies = [flat_distance_to_square(get_concept_distances(embedding_method,dataset,suffix,attributes,seed)) for seed in random_seeds]
        
    distance_list = []
    for h1,h2 in zip(baseline_embeddings,robust_hierarchies):
        distance_list.append(embedding_distance(h1,h2))
                
    return np.mean(distance_list), np.std(distance_list)
    
def robustness_image_metric(embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
    """Compute the image robustness for  metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing the average distance between each pair of images when perturbed, comparing only like-seeds
    """
    
    return compare_same_images_by_suffix(embedding_method,
                                         dataset,attributes,random_seeds,"_image_robustness",
                                        baseline_hierarchies=baseline_hierarchies)
    
def responsiveness_image_metric(embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
    """Compute the image responsiveness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
         Float, representing the average distance between each pair of images when significantly altered, comparing only like-seeds
    """

    return compare_same_images_by_suffix(embedding_method,
                                         dataset,attributes,random_seeds,"_image_responsiveness",
                                        baseline_hierarchies=baseline_hierarchies)

def robustness_model_metric(embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
    """Compute the model robustness for metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing the average distance between each pair of images when perturbed, comparing only like-seeds
    """
    
    return compare_same_images_by_suffix(embedding_method,dataset,attributes,random_seeds,"_model_robustness",
                                        baseline_hierarchies=baseline_hierarchies)


def responsiveness_model_metric(embedding_method,dataset,attributes,random_seeds,baseline_hierarchies=None):
    """Compute the model responsiveness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: String representing which dataset we're using, such as "cub"
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
         Float, representing the average distance between each pair of images when significantly altered, comparing only like-seeds
    """

    return compare_same_images_by_suffix(embedding_method,
                                         dataset,attributes,random_seeds,"_model_responsiveness",
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

def truthfulness_metric(embedding_method,dataset,attributes,random_seeds,model="VGG16",baseline_hierarchies=None):
    """Compute the truthfulness metric for a set of random seeds, given a hierarchy+embedding method
    
    Arguments: 
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

def compute_all_metrics(embedding_method,dataset,attributes,random_seeds,model="VGG16"):
    metrics = [stability_metric,robustness_image_metric,
               responsiveness_image_metric,robustness_model_metric,responsiveness_model_metric,
              truthfulness_metric]
    metric_names = ['Stability', 'Image Robustness', 
                'Image Responsiveness','Model Robustness','Model Responsiveness',
                   'Truthfulness']
            
    results = {}
    
    for metric,name in zip(metrics,metric_names):
        score = metric(embedding_method,
                        dataset,
                        attributes,
                        random_seeds)
        print("{}: {}".format(name, score))
        results[name] = score
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a concept generation method')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--dataset', type=str,
                        help='Name of the dataset used.')

    args = parser.parse_args()
    
    seeds = [43,44,45]
    
    if args.dataset == 'mnist':
        dataset = MNIST_Dataset()
    elif args.dataset == 'cub':
        dataset = CUB_Dataset()
    
    attributes = dataset.get_attributes()
    
    metrics = [truthfulness_metric,stability_metric,robustness_image_metric,
               responsiveness_image_metric,robustness_model_metric,responsiveness_model_metric]
    metric_names = ['Truthfulness','Stability', 'Image Robustness', 
                'Image Responsiveness','Model Robustness','Model Responsiveness']
        
    algorithm = args.algorithm
    if algorithm == 'labels':
        embedding_method = load_label_vectors_simple
    elif algorithm == 'tcav':
        embedding_method = load_tcav_vectors_simple
    elif algorithm == 'concept2vec':
        embedding_method = load_concept2vec_vectors_simple
    elif algorithm == 'cem':
        embedding_method = load_cem_vectors_simple
    elif algorithm == 'average':
        embedding_method = combine_embeddings_average(load_label_vectors_simple,load_tcav_vectors_simple)
    elif algorithm == 'concatenate':
        embedding_method = combine_embeddings_concatenate(load_label_vectors_simple,load_tcav_vectors_simple)
    elif algorithm == 'model':
        embedding_method = load_model_vectors_simple
    elif algorithm == 'tcav_dr':
        embedding_method = load_tcav_dr_vectors_simple
    
    results = compute_all_metrics(embedding_method,
                                    dataset,
                                    attributes,
                                    seeds)
    
    w = open("results/evaluation/{}.txt".format(args.algorithm),"w")
    for key in sorted(list(results.keys())):
        w.write("{}: {}\n".format(key,results[key]))
    w.close()
    
    
            
