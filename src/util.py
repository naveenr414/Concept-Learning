import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine 
from collections import Counter
from collections import defaultdict
import random
import tensorflow as tf
from PIL import Image
import os

def create_distance_metric(vector_a,vector_b,metric):
    """Find the distance for every pair of vectors in vector_a and vector_b
    
    Arguments:
        vector_a: nxk matrix of vectors 
        vector_b: mxk matrix of vectors
        metric: Some function, metric(a,b) -> Distance between vector a, b

    Returns:
        An nxm matrix of distances determined by metric for every pair 
            of vectors 
    """
    
    n = len(vector_a)
    m = len(vector_b)
    distances = np.zeros((n,m))
    for i,vec_a in enumerate(vector_a):
        for j,vec_b in enumerate(vector_b):
            distances[i][j] = metric(vec_a,vec_b)

    return distances

def find_average_distances(vector_list,class_list,metric = None):
    """Given a list of vectors, each a member of a certain class, find the average distance between pairs of classes
    
    Arguments:
        vector_list: n by k matrix of vectors
        class_list: int list of length n, each index representing membership in a certain class
            number 0...r-1
        metric: Which metric should be used, such as cosine, etc. 
            If it's None, we use the euclidean metric
        
    Returns:
        An rxr matrix with average distances between each class, where r=# of classes
    """
    
    if metric == None:
        distances = distance_matrix(vector_list,vector_list)
    else:
        distances = create_distance_metric(vector_list,vector_list,metric = metric)
    number_by_class = Counter(class_list)
    num_classes = len(number_by_class)
    
    avg_distance_class = np.zeros((num_classes,num_classes))
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            class_one = class_list[i]
            class_two = class_list[j]
            
            avg_distance_class[class_one][class_two] += distances[i][j]

    for i in range(num_classes):
        for j in range(num_classes):
            num_pairs = number_by_class[i]*number_by_class[j]
            if i == j:
                num_pairs = number_by_class[i]*(number_by_class[j]-1)

            if num_pairs != 0:
                avg_distance_class[i][j]/=num_pairs

    return avg_distance_class

def find_paired_distance(vector_list,class_list,random_list,metric = None):
    """Given a list of vectors, each a member of a certain class, find the average distance between pairs of classes
        Note that vectors are paired based on their 'random_concept'; we only compute distances between vectors
        which compare with the same 'random_concept', which corresponds to the baseline class in TCAV training
    
    Arguments:
        vector_list: n by k matrix of vectors
        class_list: int list of length n, each index representing membership in a certain class
            number 0...r-1
        random_list: int list of length n, each index representing which random class was used
            in the vectors computation
        metric: Which metric should be used, such as cosine, etc. 
            If it's None, we use the euclidean metric
        
    Returns:
        An rxr matrix with average distances between each class, where r=# of classes
    """
    
    if metric == None:
        metric = lambda a,b: np.linalg.norm(a-b)
        
    number_by_class = Counter(class_list)
    num_classes = len(number_by_class)
    distances_by_class = np.zeros((num_classes,num_classes))
    
    vectors_by_class_random = {}
    for i in range(num_classes):
        vectors_by_class_random[i] = {}
    
    for i in range(len(vector_list)):
        class_num = class_list[i]
        random_num = random_list[i]
        
        if random_num not in vectors_by_class_random[class_num]:
            vectors_by_class_random[class_num][random_num] = []

        vectors_by_class_random[class_num][random_num].append(vector_list[i])

    for i in range(num_classes):
        for j in range(num_classes):
            num_matches = 0
            total_distance = 0
            
            for random_num in vectors_by_class_random[i]:
                # Find all vector pairs, with (i,random_num) and (j,random_num)
                vectors_in_i = vectors_by_class_random[i][random_num]
                vectors_in_j = vectors_by_class_random[j][random_num]

                for v_i in vectors_in_i:
                    for v_j in vectors_in_j:
                        num_matches += 1
                        total_distance += metric(v_i,v_j)

            distances_by_class[i][j] = total_distance/num_matches
            
    return distances_by_class

def encode_list(class_list):
    """Encode a list of classes into a list of numbers; 
        ['dog','cat','cat','dog'] -> [0,1,1,0]
        
    Parameters:
        class_list: A list of strings, representing what each class is
    
    Returns:
        List of numbers, one for each class
    """
    
    class_to_num = {}
    for i in class_list:
        if i not in class_to_num:
            class_to_num[i] = len(class_to_num)

    return [class_to_num[i] for i in class_list]

def list_to_dict(concepts):
    """Encode a list of concepts into a dictionary, where the keys are concepts
        and the value is its position in the list
        
    Arguments: 
        concepts: A list of strings
        
    Returns: A dictionary with string keys and integer values
    """
    
    d = {}
    for i, name in enumerate(concepts):
        d[name] = i
    
    return d

def encode_list_all_concepts(concepts,all_concepts_dict):
    """Encode a list of concepts into a list of numbers, given the true ordering
    
    Arguments:
        concepts: List of string of strings, representing what each concept is
        all_concepts: List of all concepts; concepts is a subset of all_concepts
        
    Returns: List of numbers, a numeric encoding of concepts
    
    """
        
    return [all_concepts_dict[i] for i in concepts]

def find_unique_in_order(class_list):
    """Get the unique elements of a list, preserving order
        ['dog','cat','cat','dog'] -> ['dog','cat']
    
    Parameters:
        class_list: A list of strings (or any object)

    Returns:
        List containing unique elements in order
    """
    
    used = set()
    unique = [x for x in class_list if x not in used and (used.add(x) or True)]
    return unique 

def aggregate_by_class(vector_list,concept_list,aggregation_function):
    """Aggregate vector according to their membership in some class, 
        through the aggregation function
        
    Parameters:
        vector_list: Numpy matrix of concept vectors
        class_list: Membership status for each concept vector
        aggregation function: Given a list of vectors, it returns the aggregation of all vectors

    Returns:
        A list of vectors, one for each class
    """

    vector_by_class = defaultdict(lambda: [])
    for i in range(len(vector_list)):
        vector_by_class[concept_list[i]].append(vector_list[i])
    
    concept_order = find_unique_in_order(concept_list)
    ret_vector = []
    for concept in concept_order:
        ret_vector.append(aggregation_function(vector_by_class[concept]))

    return np.array(ret_vector)

def find_vectors_with_data(concept_vectors, concept_metadata, desired_metadata):
    """Find a list of concept vectors where concept metadata matches the desired metadata

    Arguments:
        concept_vectors: List of n concept vectors, as a numpy array
        concept_metadata: List of n dictionaries, with information on which concept it's for
            and which random_concept it was trained against
        desired_metadata: List of k dictionaries, with information on which vectors we want
        
    Returns:
        Numpy array of k concept vectors, corresponding to the vectors we want
    """
    
    return_vectors = []
    
    for metadata_i in desired_metadata:
        for j,metadata_j in enumerate(concept_metadata):
            for key in metadata_i:
                if metadata_j[key] != metadata_i[key]:
                    break
            else:
                return_vectors.append(concept_vectors[j])
                break

    return np.array(return_vectors)

def get_center(data):
    """Return the center of a set of n vectors in R^k
    
    Arguments: 
        Data: A numpy array of size nxk
    
    Returns: 
        A vector corresponding to the mean, of size 1xk
    """
    
    return np.mean(data,axis=0)

def load_pb(path_to_pb):
    """Load a Tensorflow Model stored in a .pb file 
    
    Arguments:
        path_to_pb: Location of the pb file, as a string
        
    Returns:
        Tensorflow graph with information on the weights of the model
    """
    
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def perturb_weights(w):
    """Given a set of weights, modify each slightly
    
    Arguments:
        w: Numpy array of weights, with different arrays having different sizes potentially
        
    Returns:
        Numpy array of weights
    """
    
    for i in range(len(w)):
        w[i] *= np.random.uniform(low=0.95,high=1.05,size=w[i].shape)
        
    return w

def responsive_weights(w):
    """Given a set of weights, modify each significantly 
    
    Arguments:
        w: Numpy array of weights, with different arrays having different sizes potentially
        
    Returns:
        Numpy array of weights
    """
    
    for i in range(len(w)):
        w[i] *= np.random.uniform(low=-1,high=1,size=w[i].shape)
        
    return w

def save_resnet_model(modification_function,output_file):
    """Save a Resnet50 model, modifying the weights according to some modification function
    
    Arguments:
        modification_function: Function that takes in a numpy array and outputs a numpy array
        output_file: Location to store the .h5 model 

    Returns: Nothing
    
    Side Effects: Saves a Resnet model
    """
    
    m = tf.keras.applications.Resnet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    
    m.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy')
    
    weights = np.array(m.get_weights())  
    new_weights = modification_function(weights)
    m.set_weights(new_weights)
    
    m.save(output_file)

def save_vgg_model(modification_function,output_file):
    """Save a Vgg16 model, modifying the weights according to some modification function
    
    Arguments:
        modification_function: Function that takes in a numpy array and outputs a numpy array
        output_file: Location to store the .h5 model 

    Returns: Nothing
    
    Side Effects: Saves a Resnet model
    """
    
    m = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    
    m.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy')
    
    weights = np.array(m.get_weights())  
    new_weights = modification_function(weights)
    m.set_weights(new_weights)
    
    m.save(output_file)
    
def resave_models():
    """Resave all 3 types of VGG models
    
    Arguments: None
    
    Returns: None
    
    Side Effects: Re-saves the three types of models
    """
    
    save_vgg_model(lambda w: w,'./dataset/models/keras/model_vgg16.h5')
    save_vgg_model(perturb_weights,'./dataset/models/keras/model_vgg16_robust.h5')
    save_vgg_model(responsive_weights,'./dataset/models/keras/model_vgg16_responsive.h5')
    
def generate_positive_examples(concept_list,num_examples):
    """For skipgram training, generate positive examples, where two concepts are co-located in the same
        example
        
    Arguments:
        concept_list: A list of numbers, each number encoding the presence of a certain concept
        num_examples: Number of positive pairs to generate
        
    Returns: Numpy array of size num_examples x 2 with positive examples
    """
    
    ret_array = []
    
    if len(set(concept_list))>1:    
        for i in range(num_examples):
            ret_array.append(random.sample(concept_list,2))

    else:
        return np.zeros((0,2))
            
    return np.array(ret_array)

def generate_negative_examples(concept_list,total_concepts, num_examples):
    """For skipgram training, generate negative examples, where one concept is selected from concept_list and 
        the other is from total_concepts (and not in concept_list

    Arguments:
        concept_list: A list of numbers, each number encoding the presence of a certain concept
        total_concepts: Total number of concepts; used to select the other concept 
        num_examples: Number of negative pairs to generate
        
        
    Returns: Numpy array of size num_examples x 2 with positive examples

    """
    
    ret_array = []
    concept_list_set = set(concept_list)
        
    for i in range(num_examples):
        example_concept = random.sample(concept_list,1)[0]
        other_concept = random.randint(0,total_concepts-1)
        while other_concept in concept_list_set and len(concept_list_set) < total_concepts:
            other_concept = random.randint(0,total_concepts-1)
            
        ret_array.append([example_concept,other_concept])
        
    return np.array(ret_array)

def generate_skipgram_dataset(datapoint,all_concepts,num_positive,num_negative):
    """ Generate a skipgram set of datapoints from a single concept list, given a list of concepts
    
    Arguments:
        datapoint: List of strings representing the presence of different concepts
        all_concepts: All potential concepts in a dataset
        num_positive: How many positive data points to generate
        num_negative: Number of negative data points to generate
    
    Returns: Numpy array of sizes (num_positive+num_negative x 2) and (num_positive+num_negative x 1) 
    """
    
    all_concepts_dict = list_to_dict(all_concepts)
    encoded_datapoint = encode_list_all_concepts(datapoint,all_concepts_dict)

    positive = generate_positive_examples(encoded_datapoint, num_positive)
    negative = generate_negative_examples(encoded_datapoint, len(all_concepts), num_negative)
    
    y_positive = np.ones((len(positive),1))
    y_negative = np.zeros((num_negative,1))
    
    whole_positive = np.concatenate([positive,y_positive],axis=1)
    whole_negative = np.concatenate([negative,y_negative],axis=1)
    
    whole_dataset = np.concatenate([whole_positive,whole_negative])
    
    np.random.shuffle(whole_dataset)
    
    return whole_dataset[:,:2], whole_dataset[:,-1]

def file_to_numpy(file_name):
    """Convert an image file to a numpy array
    
    Arguments: 
        file_name: String representing an image file

    Returns: numpy array which represents image, with pixels between 0-1
    """
    
    return np.array(Image.open(file_name)).astype("float32")/255

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

def resize_cub(images,size=64):
    """Function to resize CUB images to 64x64 for the VAE
    
    Arguments:
        images: Numpy array of images that will be resized
        
    Returns: numpy array of the resized images
    """
    
    return np.array([tf.image.resize(i,(size,size)) for i in images])
            
def contribution_score(concepts,predictions,concept_num,prediction_num,metric=False):
    """Given a concept and class we're trying to predict
        Get the impact of that concept on the classes logit
        
    Arguments:
        concepts: Numpy array of 0-1 concepts (nxk)
        prediction: Predicted logits for each data point (nxm)
        concept_num: Which concept number we're looking at (<k)
        prediciton: Which predicted class we're looking at (<m)
        
    Returns: Float for the Shapley contribution
    """
    
    positive_concept_indices = concepts[:,concept_num] == 1
    negative_concept_indices = concepts[:,concept_num] == 0
        
    logit_values = predictions[positive_concept_indices][:,prediction_num]
    logit_values_without = predictions[negative_concept_indices][:,prediction_num]
            
    if metric:
        return np.mean(logit_values)-np.mean(logit_values_without)
    else:
        return np.mean(logit_values)-np.mean(logit_values_without)
    
def save_concept_vectors(method,dataset,seed,file_name):
    """Save a list of concept vectors for a particular method to 
        results/concept_vectors/file_name.npy
        
    Arguments:
        method: One of the 'load_shapley_vectors_simple' style functions
        dataset: Object from the Dataset class
        seed: Number, indicating which seed to load
        file_name: String saying the name of the npy to be stored
    
    Returns: Nothing
    
    Side Effects: Saves all concept_vectors to results/concept_vectors/file_name.npy
    """
    
    directory = "intermediary/intervention_hierarchies/{}".format(dataset.root_folder_name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    all_vectors = []
    
    attributes = dataset.get_attributes()
    
    for a in attributes:
        vectors = method(a,dataset,"",seed=seed)
        vectors = np.mean(vectors,axis=0)
        all_vectors.append(vectors)
    all_vectors = np.array(all_vectors)
    print(all_vectors.shape)
    
    np.save(open("{}/{}.npy".format(directory,file_name),"wb"),all_vectors)
    
