import numpy as np
import pickle
import os
import shutil
from pathlib import Path
import random

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

def load_cub_split(split_name):
    """Get the information on which datapoints are used for split_name
    
    Arguments:
        split_name: Either train, val, or test
        
    Returns:
        List of Dictionaries, which contain information on data for 
            that split
    """
    
    if split_name not in ["train","val","test"]:
        raise Exception("{} not a valid split".format(split_name))

    file_name = "dataset/CUB/preprocessed/{}.pkl".format(split_name)
    data = pickle.load(open(file_name,"rb"))
    return data

def load_mnist():
    """Load the MNIST dictionary, along with concept values from train.pkl
    
    Arguments: Nothing
    
    Returns: List of dictionaries, containing info on each colored MNIST data point"""
    
    file_name = "dataset/colored_mnist/images/{}.pkl".format("train")
    return pickle.load(open(file_name,"rb"))

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

def get_mnist_images_by_attribute(attribute_name):
    """Return a list of MNIST image files with some attribute
    
    Arguments: 
        attribute_name: One of 0_color, 0_number, or spurrious

    Returns:
        String list, with the locations of each image containing attribute
    """
    
    mnist_data = load_mnist()
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i[attribute_name] == 1]
    return matching_attributes

def get_cub_images_without_attribute(attribute_name):
    """Returns a list of bird image files without some attribute
    
    Arguments:
        attribute_name: One of the 112 attributes in attributes.txt

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
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i[attribute_name] == 0]
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
    
    matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['class_label'] == random_class and i[attribute_name] == 0]
    if len(matching_attributes) == 0:
        matching_attributes = ['dataset/'+i['img_path'] for i in mnist_data if i['class_label'] == random_class] 
    
    return matching_attributes


def create_random_folder_without_attribute(attribute_name, num_folders, attribute_antifunction, images_per_folder=50):
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
    
    for folder_num in range(num_folders):
        image_locations = attribute_antifunction(attribute_name,folder_num)
        folder_location = "dataset/images/random500_{}".format(folder_num)
        if not os.path.isdir(folder_location):
            os.mkdir(folder_location)
        delete_files_in_directory(folder_location)
        
        random_images = random.sample(image_locations,images_per_folder)
        
        for image in random_images:
            shutil.copy2(image,folder_location)
    
def create_folder_from_attribute(attribute_name,attribute_function):
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
    
    image_locations = attribute_function(attribute_name)
    folder_location = "dataset/images/{}".format(attribute_name)
    if not os.path.isdir(folder_location):
        os.mkdir(folder_location)
    
    # Copy each image in image_locations to the folder
    for image in image_locations:
        shutil.copy2(image,folder_location)