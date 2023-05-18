from src.hierarchy import *
from src.concept_vectors import *
from src.dataset import *
from src.models import *
import numpy as np
import itertools
from scipy import stats
import tcav.utils as utils
from scipy.spatial.distance import cdist
import scipy
import time
import pandas as pd
from src.create_vectors import get_activations_dictionary

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
        k = 3
        
        if dataset.experiment_name == 'mnist':
            k = 1
        distance = embedding_distance(h1,h2,k=k)
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
            
    baseline_embeddings = [flat_distance_to_square(get_concept_distances
                                                   (embedding_method,dataset,"",attributes,seed)) for seed in random_seeds]
    robust_hierarchies = [flat_distance_to_square(get_concept_distances(embedding_method,dataset,suffix,attributes,seed)) for seed in random_seeds]
        
    distance_list = []
    for h1,h2 in zip(baseline_embeddings,robust_hierarchies):
        k = 3
        if dataset.experiment_name == 'mnist':
            k = 1
        distance_list.append(embedding_distance(h1,h2,k=k))
                
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
        embedding_method: A simplified embedding creation method, such as load_
cem_vectors_simple; 
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
    """Reset the images and the activations for a dataset, then redownload them, using the test dataset
        Used primarily with the truthfulness metric
    
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
                                        suffix='',num_images=max_images,train=False)
        
    
    bottlenecks = ['block4_conv1']
    # Remove existing activations
    for concept in concepts:
        activation_file_location = "{}/acts_{}_{}".format(activation_dir,concept,bottlenecks[0])
        if os.path.exists(activation_file_location):
            os.remove(activation_file_location)

def find_similar_conepts_shapley(concept,dataset,similarities,num_similar_concepts):
    """Find the num_similar_concepts most similar concepts, ranked by their similarity in 
    some layer of the Neural Network 
    
    Arguments:
        concept: Baseline concept for which we're trying to find the K most similar concepts
        dataset: String, such as 'mnist'
        similariteis: Numpy matrix with similarities 
        similar_concepts: How many similar concepts we should find 

    Returns:
        List, ordering the concepts from closest to furthest
    """
    
    attributes = dataset.get_attributes()
    concept_idx = attributes.index(concept)
    
    all_similarities = [(attributes[i],similarities[concept_idx][i]) for i in range(len(similarities)) if attributes[i] != concept]
    all_similarities = sorted(all_similarities,reverse=True,key=lambda k: k[1])[:num_similar_concepts]
    
    return [i[0] for i in all_similarities]

            
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
            
    concepts = dataset.get_attributes()
                
    our_vectors = activations[concept]
    
    all_distance_pairs = []
    for other_concept in concepts:
        if other_concept != concept:
            distances = cdist(our_vectors,activations[other_concept],metric=metric)
            all_distance_pairs.append((other_concept,np.mean(distances)))
            
    all_distance_pairs = sorted(all_distance_pairs,key=lambda k: k[1])
    return [i[0] for i in all_distance_pairs][:num_similar_concepts]

def rank_distance_concepts(all_concept_embeddings,dataset,concept,co_occuring_concepts,seed,metric='cosine'):
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
    
    attributes = dataset.get_attributes()
    concept_const = attributes.index(concept)
    
    embedding_constant = all_concept_embeddings[concept_const]
    embedding_by_concept = {}

    for other_concept in co_occuring_concepts:
        embedding_by_concept[other_concept] =  all_concept_embeddings[attributes.index(other_concept)]
        
    metric_concept_pairs = [(other_concept,
                             np.mean(cdist(embedding_constant,embedding_by_concept[other_concept],metric=metric)))
                             for other_concept in co_occuring_concepts]

    metric_concept_pairs = sorted(metric_concept_pairs, key=lambda k: k[1])
    return [i[0] for i in metric_concept_pairs]



def get_model_concept_similarities(dataset,model):
    """Given a dataet + Keras model, run the model over the validation and get 
        similarities between concepts via Shapley method

    Arguments:
        dataset: Object from dataset class
        model: Keras model
        
    Returns: Numpy square matrix of similarities between concepts

    """
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    image_size = (224, 224)
    
    num_classes = len(set([i['class_label'] for i in dataset.get_data()]))
    num_attributes = len(dataset.get_data()[0]['attribute_label'])
    
    data_valid = dataset.get_data(train=False)
    img_paths_valid = ['dataset/'+i['img_path'] for i in data_valid]
    labels_valid = [str(i['class_label']) for i in data_valid]
    valid_df = pd.DataFrame(zip(img_paths_valid,labels_valid), columns=["image_path", "label"])

    valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                              x_col="image_path",
                                              y_col="label",
                                              target_size=image_size,
                                              batch_size=batch_size,
                                              class_mode="categorical",
                                              shuffle=False)
    
    predictions = model.predict(valid_generator)

    concepts = np.array([i['attribute_label'] for i in data_valid])
    contribution_array = np.array([[contribution_score(concepts,predictions,concept_num,class_num,metric=True) for class_num in range(num_classes)]  for concept_num in range(num_attributes)])    
    
    dist_array = cdist(contribution_array, contribution_array, metric='cosine')
    
    similarity_array = 1-dist_array
    
    return similarity_array
    
    
def truthfulness_metric_shapley(embedding_method,dataset,attributes,random_seeds,model_name="VGG16",baseline_hierarchies=None,compare_concepts=None):
    """Compute the truthfulness metric using a Shapley-like approach, given a hierarchy+embedding method

    Arguments: 
        embedding_method: A simplified embedding creation method, such as load_cem_vectors_simple; 
            Simply loads embeddings, does not train them from scratch 
        dataset: Object from the dataset class
        attributes: List of attributes we want to create embeddings for
        random_seeds: List of numbers representing the random seed for the embeddings

    Returns:
        Float, representing similarity between distances in the model, and the distances predicted by the hierarchy; it's an average correlation between 0-1, and the standard deviation
    """
    
    if compare_concepts == None:
        compare_concepts = 5   
    
    # For MNIST, only the closest concept matters
    if dataset.experiment_name == "mnist":
        compare_concepts = 1
    
    num_classes = len(set([i['class_label'] for i in dataset.get_data()]))
    
    avg_truthfulness = []
    
    model = get_large_image_model(dataset,model_name)
    model.load_weights("results/models/{}_models/{}_42.h5".format(model_name.lower(),dataset.experiment_name))
    
    similarity_matrix = get_model_concept_similarities(dataset,model)
    
    for seed in random_seeds:  
        random.seed(seed)
        np.random.seed(seed)
        
        temp_truthfulness = []
        
        all_concept_embeddings = np.array([embedding_method(i,dataset,"",seed=seed) for i in dataset.get_attributes()])
        
        for concept in dataset.get_attributes():
            co_occuring_concepts = find_similar_conepts_shapley(concept,dataset,similarity_matrix,compare_concepts)
            co_occuring_concepts_hierarchy = rank_distance_concepts(all_concept_embeddings,dataset,concept,
                                                                    attributes,seed)[:1+compare_concepts]
            co_occuring_concepts_hierarchy = [i for i in co_occuring_concepts_hierarchy if i!=concept]
            co_occuring_concepts_hierarchy = co_occuring_concepts_hierarchy[:compare_concepts]
            
            """
            
            if len(co_occuring_concepts) == 1:
                temp_truthfulness.append(int(co_occuring_concepts == co_occuring_concepts_hierarchy))
            else:
                temp_truthfulness.append(stats.kendalltau(co_occuring_concepts,
                                                         co_occuring_concepts_hierarchy).correlation)"""
            
            intersection = set(co_occuring_concepts).intersection(set(co_occuring_concepts_hierarchy))
            temp_truthfulness.append(len(intersection)/compare_concepts)
            
        avg_truthfulness.append(np.mean(temp_truthfulness))
            
        
    return np.mean(avg_truthfulness), np.std(avg_truthfulness)

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
    compare_concepts = 5
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
        all_concept_embeddings = np.array([embedding_method(i,dataset,"",seed=seed) for i in dataset.get_attributes()])
        
        temp_truthfulness = []
        
        for concept in selected_concepts:
            co_occuring_concepts = find_similar_concepts(concept,dataset,activations,compare_concepts,seed)                        
            co_occuring_concepts_hierarchy = rank_distance_concepts(all_concept_embeddings,dataset,concept,
                                                                    attributes,seed)[1:1+compare_concepts]
            
            intersection = set(co_occuring_concepts).intersection(set(co_occuring_concepts_hierarchy))
            temp_truthfulness.append(len(intersection)/compare_concepts)            
        avg_truthfulness.append(np.mean(temp_truthfulness))
            
        
    return np.mean(avg_truthfulness), np.std(avg_truthfulness)

def compute_all_metrics(embedding_method,dataset,attributes,random_seeds,model="VGG16",eval_truthfulness=True,metrics=[],metric_names=[]):
    if metrics == []:
        metrics = [stability_metric, robustness_image_metric,responsiveness_image_metric]
        metric_names = ['Stability', 'Image Robustness', 'Image Responsiveness']
            
    results = {}
    for metric,name in zip(metrics,metric_names):
        score = metric(embedding_method,
                        dataset,
                        attributes,
                        random_seeds)
        print("{}: {}".format(name, score))
        results[name] = score
    
    if eval_truthfulness:
        results['Truthfulness'] = truthfulness_metric_shapley(embedding_method,
                        dataset,
                        attributes,
                        random_seeds)
        print("{}: {}".format("Truthfulness",results['Truthfulness']))
        
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
    
    model = 'VGG16'
        
    algorithm = args.algorithm
    if algorithm == 'labels':
        embedding_method = load_label_vectors_simple
    elif algorithm == 'tcav':
        embedding_method = load_tcav_vectors_simple
    elif algorithm == 'concept2vec':
        embedding_method = load_concept2vec_vectors_simple
    elif algorithm == 'cem':
        embedding_method = load_cem_vectors_simple
        model='Resnet50'
    elif algorithm == 'cem_concept':
        embedding_method = load_cem_loss_vectors_simple
        model = 'Resnet50'
    elif algorithm == 'average':
        embedding_method = combine_embeddings_average(load_label_vectors_simple,load_tcav_vectors_simple)
    elif algorithm == 'concatenate':
        embedding_method = combine_embeddings_concatenate(load_label_vectors_simple,load_tcav_vectors_simple)
    elif algorithm == 'model':
        embedding_method = load_model_vectors_simple
    elif algorithm == 'tcav_dr':
        embedding_method = load_tcav_dr_vectors_simple
    elif algorithm == 'vae':
        embedding_method = load_vae_vectors_simple
    elif algorithm == 'vae_concept':
        embedding_method = load_vae_concept_vectors_simple

    results = compute_all_metrics(embedding_method,
                                    dataset,
                                    attributes,
                                    seeds,
                                 model=model)
    
    w = open("results/evaluation/{}.txt".format(args.algorithm),"w")
    for key in sorted(list(results.keys())):
        w.write("{}: {}\n".format(key,results[key]))
    w.close()