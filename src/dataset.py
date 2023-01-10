import numpy as np
import pickle
import os
import shutil
from pathlib import Path
import random
import glob
from PIL import Image
from copy import deepcopy

def add_gaussian_noise(img_array,mean,standard_deviation):
    """Given a numpy array representing an image, add gaussian noise to the array
    
    Arguments:
        img_array: Numpy array with values between 0-255
        mean: Mean for the Gaussian noise
        standard_deviation: Standard Deviation (Sigma) for the standard deviation

    Returns:
        numpy array with random noise
    """
    
    noise = np.random.normal(mean,standard_deviation,img_array.shape)
    result = img_array + noise
    result = np.clip(result,0,255).astype(int)
    
    return result

def create_junk_image(img_array):
    """Given a numpy array representing an image, return an image of the same size with random values
    
    Arguments:
        img_array: Numpy array with values between 0-255
        
    Returns: 
        numpy array with junk values
        
    """
    
    output_arr = np.random.uniform(0,255,img_array.shape)
    return output_arr.astype(int)

def write_new_image_function(input_path,output_path,func):
    """Create a new image by applying some function to an input image
    
    Arguments:
        input_path: Location of image
        output_path: Location to where the output image should be written
        func: Function that takes in a numpy array and outputs another numpy array
        
    Returns: Nothing
    
    Side Effects: Re-writes image at output_path
    """
    
    im1 = Image.open(input_path)
    new_arr = func(np.array(im1)).astype(np.uint8)
        
    output_image = Image.fromarray(new_arr)
    output_image.save(output_path)
    
def run_function_MNIST(output_folder_name,func):
    """Run some function over all MNIST files, based on what the output folder should be called
    
    Arguments: 
        output_folder_name: Striong representing which folder everything should be written to, like colored_mnist_robustness
        func: Some function that takes in an image_array and outputs an image_array
    
    Returns: Nothing
    
    Side Effects: Runs some function over all images in MNIST, and saves it in output_folder
    """
    
    all_input_files = glob.glob("dataset/colored_mnist/images/*/*.png")
    
    for input_file in all_input_files:
        corresponding_output = input_file.replace("colored_mnist",output_folder_name)
        write_new_image_function(input_file,corresponding_output,func)
        
def create_gaussian_MNIST():
    """Create the robustness MNIST dataset by running Gaussian Noise over all MNIST images
    
    Arguments: Nothing
    
    Returns: Nothing
    
    Side Effects: Adds Gaussian Noise to all images in colored_mnist_robustness
    """
    
    flip_probability = .01    
    for file_name in ["train","val"]:
        flip_concept_labels_file("dataset/colored_mnist/images/{}.pkl".format(file_name),
                                 "dataset/colored_mnist_robustness/images/{}.pkl".format(file_name),
                                 flip_probability,
                                 lambda s: s.replace("colored_mnist","colored_mnist_robustness"))
    
    run_function_MNIST("colored_mnist_robustness",lambda arr: add_gaussian_noise(arr,0,50))
    
def create_junk_MNIST():
    """Create the responsiveness MNIST dataset by creating Junk Images over all MNIST images
    
    Arguments: Nothing
    
    Returns: Nothing
    
    Side Effects: Adds Gaussian Noise to all images in colored_mnist_robustness
    """
    
    flip_probability = 0.5
    
    for file_name in ["train","val"]:
        flip_concept_labels_file("dataset/colored_mnist/images/{}.pkl".format(file_name),
                                 "dataset/colored_mnist_responsiveness/images/{}.pkl".format(file_name),
                                 flip_probability,
                                 lambda s: s.replace("colored_mnist","colored_mnist_responsiveness"))

    run_function_MNIST("colored_mnist_responsiveness", create_junk_image)

        
def flip_concept_labels(concept_list,flip_prob,img_path_update):
    """Flip probabilities of concept labels according to some probability
    
    Arguments:
        concept_list: List of dictionaries representing concept labels
        flip_prob: The probability any individual concept is flipped
        img_path_update: Function to update the images path when flipping concept labels
        
    Returns:
        New concept_list with concepts flipped 
    """
    
    new_arr = deepcopy(concept_list)
    
    for data in new_arr:
        data['img_path'] = img_path_update(data['img_path'])
        
        for i in range(len(data['attribute_label'])):
            if np.random.random() < flip_prob:
                data['attribute_label'][i] = 1-data['attribute_label'][i]
                
    return new_arr

def flip_concept_labels_file(input_file,output_file,flip_probability,img_path_update):
    """Given an input file with concept labels, such as train.pkl, and an output file, flip attribute_labels with 
        probability flip_probability
        
    Arguments:
        input_file: train.pkl or valid.pkl which stores a list of dictionaries
        output_file: Place to put the new pkl file
        flip_probability: Probability of independently flipping any attribute from 0<->1
        img_path_update: Function to update image paths
        
    Returns: Nothing
    
    Side Effects: Writes a new flipped pkl file to output_file
    """
    
    all_concepts = pickle.load(open(input_file,"rb"))
    new_all_concepts = flip_concept_labels(all_concepts,flip_probability,img_path_update)
    pickle.dump(new_all_concepts,open(output_file,"wb"))
        

def get_seed_numbers(folder_location):
    """Get all sub-directory names in a folder; used typically to find which seeds were used for an experiment
    
    Arguments: 
        folder_location: String representing hte location of the directory
        
    Returns: 
        List of Strings, representing sub-folders
    """
    all_seeds = glob.glob(folder_location+"/*")
    
    return [i.split("/")[-1] for i in all_seeds]
    
def delete_files_in_directory(directory_name):
    """Delete all files in a directory (non-recursive)
    
    Arguments: 
        directory_name: Location of directory, where files are to be deleted
    
    Returns: none
    
    Side Effects:
        Deletes all files in folder directory_name
    """
    
    [f.unlink() for f in Path(directory_name).glob("*") if f.is_file()] 


def get_cub_attributes():
    """Get the list of attributes used in CUB classification
    These are stored in the attributes.txt file
    
    Arguments: None
    
    Returns: String list of attributes
    """
    
    attributes_file = "dataset/CUB/metadata/attributes.txt"
    lines = open(attributes_file).read().strip().split("\n")
    attributes = [i.split(" ")[1] for i in lines]
    return attributes

def get_mnist_attributes():
    """Get the lit of attributes used in Colored MNIST
    This is 0_color, 0_number... and spurious 
    
    Arguments: None
    
    Returns: String list of attributes
    """
    
    attributes = []
    
    for i in range(10):
        attributes += ["{}_color".format(i),"{}_number".format(i)]

    attributes += ["spurious"]
    return attributes

def load_cub_split(split_name,seed=-1):
    """Get the information on which datapoints are used for split_name
    
    Arguments:
        split_name: Either train, val, or test
        seed: Optional random number that sets the order of data
        
    Returns:
        List of Dictionaries, which contain information on data for 
            that split
    """
    
    if split_name not in ["train","val","test"]:
        raise Exception("{} not a valid split".format(split_name))

    file_name = "dataset/CUB/preprocessed/{}.pkl".format(split_name)
    data = pickle.load(open(file_name,"rb"))
    
    if seed>-1:
        random.seed(seed)
    random.shuffle(data)
    
    return data

def load_mnist(seed=-1,suffix=''):
    """Load the MNIST dictionary, along with concept values from train.pkl
    
    Arguments: Seed: Optional number that changes the order of data
        Suffix: Optional, potential variant of the MNIST dataset
    
    Returns: List of dictionaries, containing info on each colored MNIST data point"""
    
    file_name = "dataset/colored_mnist{}/images/{}.pkl".format("train",suffix)
    data = pickle.load(open(file_name,"rb"))
    
    if seed > -1:
        random.seed(seed)
    random.shuffle(data)
    return data
    
def get_cub_images_by_attribute(attribute_name):
    """Return a list of bird image files with some attribute
    
    Arguments: 
        attribute_name: One of the 112 attributes in attributes.txt

    Returns:
        String list, with the locations of each image containing attribute
    """
    
    all_attributes = get_cub_attributes()
    if attribute_name not in all_attributes:
        raise Exception("{} not found in the 112 CUB attributes".format(attribute_name))
        
    attribute_index = all_attributes.index(attribute_name)
    train_data = load_cub_split("train")
    
    unfiltered_image_locations = [i['img_path'] for i in train_data if i['attribute_label'][attribute_index] == 1]
    filtered_image_locations = [i.replace("/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/","") 
                                 for i in unfiltered_image_locations]
    prefix = "dataset/CUB/images/"
    filtered_image_locations = [prefix+i for i in filtered_image_locations]
    return filtered_image_locations

def get_cub_classes_by_attribute(attribute_name):
    """Return a list of CUB bird types that have a particular attribute
    Note that, because of the CUB data collection method, all birds in a particular class
    have the same attributes
    
    Arguments: 
        attribute_name: One of the 112 attributes.txt
        
    Returns:
        String list, with the name of bird types
    """
    
    image_locations = get_cub_images_by_attribute(attribute_name)
    bird_types = [i.split("/")[5].split(".")[1].replace("_"," ") for i in image_locations]
    
    return sorted(list(set(bird_types)))
    

def get_mnist_images_by_attribute(attribute_name):
    """Return a list of MNIST image files with some attribute
    
    Arguments: 
        attribute_name: One of 0_color, 0_number, or spurrious

    Returns:
        String list, with the locations of each image containing attribute
    """
    
    mnist_data = load_mnist()
    mnist_attributes = get_mnist_attributes()
    attribute_index = mnist_attributes.index(attribute_name)
    
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['attribute_label'][attribute_index] == 1]
    return matching_attributes

def get_cub_images_without_attribute(attribute_name,folder_num=0):
    """Returns a list of bird image files without some attribute
    
    Arguments:
        attribute_name: One of the 112 attributes in attributes.txt
        folder_num: Optional, unused parameter which can allow us to specify which images without attribute we're taking

    Returns:
        String list, with the locations of each image without the attribute
    """
    
    all_attributes = get_cub_attributes()
    train_data = load_cub_split("train")
    all_image_locations = [i['img_path'].replace("/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/","") 
                                 for i in train_data]
    prefix = "dataset/CUB/images/"
    all_image_locations = [prefix+i for i in all_image_locations]
    
    locations_with_attribute = get_cub_images_by_attribute(attribute_name)
    locations_with_attribute = set(locations_with_attribute)
    
    locations_without_attribute = [i for i in all_image_locations if i not in locations_with_attribute]
    return locations_without_attribute

def get_mnist_images_without_attribute(attribute_name,folder_num=0):
    """Return a list of MNIST image files without some attribute
    
    Arguments: 
        attribute_name: One of 0_color, 0_number, or spurrious

    Returns:
        String list, with the locations of each image lacking the attribute
    """
    
    mnist_data = load_mnist()
    mnist_attributes = get_mnist_attributes()
    
    attribute_index = mnist_attributes.index(attribute_name)
    
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['attribute_label'][attribute_index] == 0]
    return matching_attributes

def get_mnist_images_without_attribute_one_class(attribute_name,folder_num):
    """Return a list of MNIST image files without some attribute, 
        where all images are from one class
    
    Arguments: 
        attribute_name: One of 0_color, 0_number, or spurrious

    Returns:
        String list, with the locations of each image lacking the attribute
    """
    
    mnist_data = load_mnist()
    
    # Find some random class without the attribute
    random_class = folder_num % 10 
    
    mnist_attributes = get_mnist_attributes()
    attribute_index = mnist_attributes.index(attribute_name)
    
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['class_label'] == random_class and i['attribute_label'][attribute_index] == 0]
    if len(matching_attributes) == 0:
        matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['class_label'] == random_class] 
    
    return matching_attributes


def create_random_folder_without_attribute(attribute_name, num_folders, attribute_antifunction, images_per_folder=50,seed=-1):
    """Create new folders, with each folder containing birds without a particular attribute
    
    Arguments:
        attribute_name: String that's one of the 112 CUB attributes
        num_folders: Number of folders to create; this is the 
            same as the num_random_exp variable
        attribute_antifunction: Function that returns image locations for all iamges
            lacking a particular attribute
            Either get_cub_images_without_attribute or get_mnist_images_without_attribute
        images_per_folder: How many images are in each folder
    
    Returns:
        None
        
    Side Effects:
        Creates num_folders new folders, populated with images
            lacking a particular attribute
    """
    
    if seed > -1:
        random.seed(seed)
    
    for folder_num in range(num_folders):
        image_locations = attribute_antifunction(attribute_name,folder_num)
        folder_location = "dataset/images/random500_{}".format(folder_num)
        if not os.path.isdir(folder_location):
            os.mkdir(folder_location)
        delete_files_in_directory(folder_location)
        
        random_images = random.sample(image_locations,images_per_folder)
        
        for image in random_images:
            shutil.copy2(image,folder_location)
    
def create_folder_from_attribute(attribute_name,attribute_function,seed=-1):
    """Create a new folder, with the images being all birds with a particular attribute
    
    Arguments:
        attribute_name: String that's an attribute to either the MNIST dataset or the Birds Dataset
        attribute_function: Function that retrieves all matching images, given an attribute
            Either get_mnist_images_by_attribute or get_cub_images_by_attribute

    Returns:
        None
        
    Side Effects:
        Creates a new folder, populated with images
            with a particular attribute
    """
    
    if seed > -1:
        random.seed(seed)
    
    image_locations = attribute_function(attribute_name)
    folder_location = "dataset/images/{}".format(attribute_name)
    if not os.path.isdir(folder_location):
        os.mkdir(folder_location)
    
    # Copy each image in image_locations to the folder
    for image in image_locations:
        shutil.copy2(image,folder_location)