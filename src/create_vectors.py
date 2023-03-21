import tcav
import numpy as np
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model as model
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
from src.models import *
from src.util import *
import keras
import time
from sklearn.decomposition import PCA
import pandas as pd 
import os

class ResnetWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])
    
class VGGWrapper(model.KerasModelWrapper):
    def get_image_shape(self):
        return np.array([224,224,3])

def create_concept2vec(dataset,suffix,seed=-1,
                             embedding_size=32,num_epochs=5,dataset_size=1000,initial_embedding=None):
    """Generate concept2vec vectors by training a Skipgram architecture on correlated concepts
    
    Arguments:
        dataset: Object from the Dataset Class
        suffix: String that represents a specific variant of the dataset
        seed: Number for the random seed
        embedding_size: Size of the embeddings; by default it's 32
        
    Returns: Nothing
    
    Side Effects: Trains a set of vectors at results/concept2vec/experimentname+suffix/seed/
    """
    
    if seed == -1:
        seed = random.randint(0,100000)
    
    destination_folder = "results/concept2vec/{}/{}".format(dataset.experiment_name+suffix,seed)
    
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    attributes = dataset.get_attributes()
    V = len(attributes)
    all_data = dataset.get_data(seed,suffix)[:dataset_size]
    
    SkipGram = create_skipgram_architecture(embedding_size,V,initial_embedding=initial_embedding)
    
    for _ in range(num_epochs):
        X = np.zeros((0,2),dtype=np.int32)
        Y = np.zeros((0,))

        # Collect the dataset
        for i, doc in enumerate(all_data):
            formatted_data = [attributes[j] for j,indicator in enumerate(doc['attribute_label']) if indicator == 1]
            data, labels = generate_skipgram_dataset(formatted_data,attributes,8,8)   
            
            data = np.array(data,dtype=np.int32)

            X = np.concatenate([X,data])
            Y = np.concatenate([Y,labels])
            
        X_0 = X[:,0]
        X_1 = X[:,1]
        
        SkipGram.fit([X_0,X_1],Y)
    
    vectors = SkipGram.get_weights()[0]
    
    np.save(open("{}/vectors.npy".format(destination_folder),"wb"),vectors)
    
def create_tcav_dataset(attribute_name,dataset,num_random_exp,
                        max_examples=100,images_per_folder=50,seed=-1,suffix='',model_name="VGG16",bottlenecks=["block4_conv1"]):
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
            GRAPH_PATH = "./dataset/models/keras/model_resnet.h5"
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
                               experiment_name="unfiled",max_examples=500,bottleneck="block4_conv1",delete_activations=True):
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
    
    start = time.time()
    
    if model_name == "Resnet50":
        bottleneck = "conv4_block6_out"

    
    if delete_activations:
        delete_previous_activations(bottleneck,attribute_list)

    start = time.time()    
    act_generator = load_activations_model(experiment_name,max_examples,model_name,sess)
            
    acts = {}
    for i in attribute_list:
        start = time.time()
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
        
def create_model_vectors(attributes,dataset,suffix,seed=-1):
    """Develop concept vectors based on a model representation
    
    Arguments:
        attributes: List of attributes to create concept vectors for
        dataset: Object from the dataset class
        suffix: String representing which specific instance of the dataset we're testing
        seed: Random number seed
        
    Returns: Nothing
    
    Side Effects: Saves concpet vectors to results/model_vectors/seed/attribute.npy"""
    
    
    max_images = 25
    model = "VGG16"
    
    if suffix == "_model_robustness":
        model = "VGG16Robustness"
    elif suffix == "_model_responsiveness":
        model = "VGG16Responsiveness"
        
    for attribute in attributes:
        create_folder_from_attribute(attribute,
                                     dataset.get_images_with_attribute,num_images=max_images,suffix=suffix,seed=seed)
    
    with tf.compat.v1.Session() as sess:
        activation_generator = load_activations_model(dataset.experiment_name+suffix,max_images,model,sess)
        activations = get_activations_dictionary(attributes,
                                                 sess,
                                                 model_name=model,
                                                 experiment_name=dataset.experiment_name,
                                                 max_examples=max_images)
    
    if not os.path.exists("results/model_vectors/{}/{}".format(dataset.experiment_name+suffix,seed)):
        os.makedirs("results/model_vectors/{}/{}".format(dataset.experiment_name+suffix,seed))
        
    for attribute in activations:
        save_file = "results/model_vectors/{}/{}/{}.npy".format(dataset.experiment_name+suffix,seed,attribute)
        np.save(open(save_file,"wb"),activations[attribute])

def create_shapley_vectors(attributes,dataset,suffix,seed=-1):
    """Develop concept vectors based on shapley impact on logits
    
    Arguments:
        attributes: List of attributes to create concept vectors for
        dataset: Object from the dataset class
        suffix: String representing which specific instance of the dataset we're testing
        seed: Random number seed
        
    Returns: Nothing
    
    Side Effects: Saves concpet vectors to results/shapley/seed/attribute.npy"""

    data = dataset.get_data(train=True)
    img_paths = ['dataset/'+i['img_path'] for i in data]
    labels = [str(i['class_label']) for i in data]

    num_attributes = len(attributes)
    num_classes = len(set(labels))
    
    concept_vectors = {}
    model_name = "VGG16"
    model = get_large_image_model(dataset,model_name)
    model.load_weights("results/models/{}_models/{}_{}.h5".format(model_name.lower(),dataset.experiment_name+suffix,seed))
    
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    image_size = (224, 224)
    
    valid_df = pd.DataFrame(zip(img_paths,labels), columns=["image_path", "label"])

    valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                              x_col="image_path",
                                              y_col="label",
                                              target_size=image_size,
                                              batch_size=batch_size,
                                              class_mode="categorical",
                                              shuffle=False)
    
    predictions = model.predict(valid_generator)
    
    concepts = np.array([i['attribute_label'] for i in data[:len(predictions)]])
    contribution_array = np.array([[contribution_score(concepts,predictions,concept_num,class_num) 
                                    for class_num in range(num_classes)] for concept_num in range(num_attributes)])
    
    for i in range(len(attributes)):
        concept_vectors[attributes[i]] = contribution_array[i].reshape((1,num_classes))
    
    if not os.path.exists("results/shapley/{}/{}".format(dataset.experiment_name+suffix,seed)):
        os.makedirs("results/shapley/{}/{}".format(dataset.experiment_name+suffix,seed))
        
    for attribute in attributes:
        save_file = "results/shapley/{}/{}/{}.npy".format(dataset.experiment_name+suffix,seed,attribute)
        np.save(open(save_file,"wb"),concept_vectors[attribute])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors from a dataset')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate concept vectors')
    parser.add_argument('--dataset',type=str,
                        help='Which dataset to generate vectors for, such as mnist or cub')
    parser.add_argument('--suffix',type=str,
                        help='Which subset of the dataset to train on, such image_robustness, etc, if nothing then "none"',
                       default='none') 
    parser.add_argument('--seed',type=int, default=42,
                    help='Random seed used in tcav experiment')
    
    
    # TCAV Arguments
    parser.add_argument('--class_name', type=str, default='',
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

    args = parser.parse_args()
    
    if args.suffix == 'none':
        args.suffix = ''
    else:
        args.suffix = '_'+args.suffix
        
    if args.dataset not in ['xor','mnist','imagenet','cub']:
        raise Exception("{} dataset not supported".format(args.dataset))
        
    if args.dataset == 'mnist':
        dataset = MNIST_Dataset()
    elif args.dataset == 'cub':
        dataset = CUB_Dataset()
        
    if args.algorithm == 'tcav':
        if args.dataset == 'imagenet':
            create_tcav_vectors([args.class_name],args.target,args.model_name[args.bottleneck],
                                args.num_random_exp,alphas=[args.alpha],seed=args.seed)
        else:        
            create_tcav_dataset(args.class_name,dataset,
                                args.num_random_exp,args.images_per_folder,
                                seed=args.seed,suffix=args.suffix,model_name=args.model_name,bottlenecks=[args.bottleneck])
    elif args.algorithm == 'model':
        attributes = dataset.get_attributes()
        create_model_vectors(attributes,dataset,args.suffix,args.seed)
    elif args.algorithm == 'shapley':
        attributes = dataset.get_attributes()
        create_shapley_vectors(attributes,dataset,args.suffix,args.seed)
    else:
        raise Exception("{} not implemented to generate concept vectors".format(args.algorithm))