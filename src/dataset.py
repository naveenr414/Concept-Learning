import numpy as np
import pickle
import os
import shutil
from pathlib import Path
import random
import glob
from PIL import Image
from copy import deepcopy
class Dataset:
    def get_attributes(self):
        pass
    
    def get_data(self,seed=-1,suffix=""):
        if suffix == "_model_robustness" or suffix == "_model_responsiveness":
            suffix = ""
        
        file_name = self.pkl_path.format(suffix)
        data = pickle.load(open(file_name,"rb"))

        if seed > -1:
            random.seed(seed)
        random.shuffle(data)
        return data
    
    def get_class_labels(self,seed=-1,suffix=""):
        data = self.get_data(seed,suffix)
        return set([i['class_label'] for i in data])
    
    def get_images_with_attribute(self,attribute_name,seed=-1,suffix=""):
        data = self.get_data(seed,suffix)
        attributes = self.get_attributes()
        attribute_index = attributes.index(attribute_name)

        matching_attributes = [self.path_to_image(i['img_path']) 
                               for i in data if i['attribute_label'][attribute_index] == 1]
        return matching_attributes
    
    def get_images_without_attribute(self,attribute_name,one_class=False,seed=-1,suffix=""):
        data = self.get_data(seed,suffix)
        attributes = self.get_attributes()
        attribute_index = attributes.index(attribute_name)
        
        if one_class:
            all_classes = self.get_class_labels(seed,suffix)
            random_class = random.sample(list(all_classes),k=1)[0]

            matching_attributes = [self.path_to_image(i['img_path']) 
                                   for i in data if i['class_label'] == random_class and 
                                   i['attribute_label'][attribute_index] == 0]
            if len(matching_attributes) == 0:
                matching_attributes = [self.path_to_image(i['img_path']) 
                                       for i in data if i['class_label'] == random_class] 

        else:
            matching_attributes = [self.path_to_image(i['img_path']) 
                                   for i in data if i['attribute_label'][attribute_index] == 0]
        
        return matching_attributes


    
    def create_gaussian(self):
        flip_probability = .01
        input_file = self.pkl_path.format('')
        output_file = self.pkl_path.format('_image_robustness')
        flip_concept_labels_file(input_file,
                                 output_file,
                                 flip_probability,
                                 lambda s: s.replace(self.root_folder_name,
                                                     self.root_folder_name+"_image_robustness"))

        self.run_function(self.root_folder_name+"_image_robustness",lambda arr: add_gaussian_noise(arr,0,50))
    
    def create_junk(self):
        flip_probability = 0.5
        input_file = self.pkl_path.format('')
        output_file = self.pkl_path.format('_image_responsiveness')

        flip_concept_labels_file(input_file,
                                 output_file,
                                 flip_probability,
                                 lambda s: s.replace(self.root_folder_name,
                                                     self.root_folder_name+"_image_responsiveness"))
        self.run_function(self.root_folder_name+"_image_robustness",lambda arr: create_junk_image)
    
    def run_function(self,output_folder_name,func):
        all_input_files = glob.glob(self.all_files)

        for input_file in all_input_files:
            corresponding_output = input_file.replace(self.root_folder_name,output_folder_name)
            write_new_image_function(input_file,corresponding_output,func)

    
class MNIST_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = "dataset/colored_mnist{}/images/train.pkl"
        self.path_to_image = lambda path: "dataset/"+path
        self.all_files = "dataset/colored_mnist/images/*/*.png"
        self.root_folder_name = "colored_mnist"
        self.experiment_name = "mnist"
        
    def get_attributes(self):
        """Get the information on which datapoints are used for split_name
    
        Arguments:
            split_name: Either train, val, or test
            seed: Optional random number that sets the order of data

        Returns:
            List of Dictionaries, which contain information on data for 
                that split
        """
        
        attributes = []
    
        for i in range(10):
            attributes += ["{}_color".format(i),"{}_number".format(i)]

        attributes += ["spurious"]
        return attributes

class CUB_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = "dataset/CUB{}/preprocessed/train.pkl"
        self.path_to_image = lambda path: "dataset/CUB/images/" + path.replace("/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/","")
        self.all_files = "dataset/CUB/images/CUB_200_2011/images/*/*.jpg"
        self.root_folder_name = "CUB"
        self.experiment_name = "cub"
    
    def get_attributes(self):
        attributes_file = "dataset/CUB/metadata/attributes.txt"
        lines = open(attributes_file).read().strip().split("\n")
        attributes = [i.split(" ")[1] for i in lines]
        return attributes

class XOR_Dataset(Dataset):
    def __init__(self):
        self.experiment_name = "xor"
    def get_attributes(self):
        return ['0','1']
    
class Unfiled_Dataset(Dataset):
    def __init__(self):
        self.experiment_name = "unfiled"


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


def create_random_folder_without_attribute(attribute_name, num_folders, attribute_antifunction,
                                           suffix='',images_per_folder=50,seed=-1):
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
        image_locations = attribute_antifunction(attribute_name,suffix=suffix,seed=seed)
        folder_location = "dataset/images/random500_{}".format(folder_num)
        if not os.path.isdir(folder_location):
            os.mkdir(folder_location)
        delete_files_in_directory(folder_location)
                
        random_images = random.sample(image_locations,images_per_folder)
        
        for image in random_images:
            shutil.copy2(image,folder_location)
    
def create_folder_from_attribute(attribute_name,attribute_function,seed=-1,suffix='',num_images=100):
    """Create a new folder, with the images being all birds with a particular attribute
    
    Arguments:
        attribute_name: String that's an attribute to either the MNIST dataset or the Birds Dataset
        attribute_function: Function that retrieves all matching images, given an attribute
            Either get_mnist_images_by_attribute or get_cub_images_by_attribute
        num_images: How many images to select from the attribute folder, at most

    Returns:
        None
        
    Side Effects:
        Creates a new folder, populated with images
            with a particular attribute
    """
    
    if seed > -1:
        random.seed(seed)
    
    image_locations = attribute_function(attribute_name,suffix=suffix,seed=seed)
    
    folder_location = "dataset/images/{}".format(attribute_name)
    if not os.path.isdir(folder_location):
        os.mkdir(folder_location)
        
    # Clear that folder 
    files = glob.glob("{}/*".format(folder_location))
    for f in files:
        os.remove(f)

    if len(image_locations)>num_images:
        image_locations = random.sample(image_locations,k=num_images)
    
    # Copy each image in image_locations to the folder
    for image in image_locations:
        shutil.copy2(image,folder_location)