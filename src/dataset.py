import numpy as np
import pickle
import os
import shutil
from pathlib import Path
import random
import glob
from PIL import Image
from copy import deepcopy

dataset_folder = "../../cem/cem"

class Dataset:
    def get_attributes(self):
        pass
    
    def get_data(self,seed=-1,suffix="",train=True):
        if suffix == "_model_robustness" or suffix == "_model_responsiveness":
            suffix = ""
            
        if train:
            file_name = self.pkl_path.format(suffix)
        else:
            file_name = self.test_pkl_path.format(suffix)
        data = pickle.load(open(file_name,"rb"))
                
        if seed > -1:
            random.seed(seed)
            random.shuffle(data)
        return data
    
    def get_class_labels(self,seed=-1,suffix="",train=True):
        data = self.get_data(seed,suffix,train=train)
        return set([i['class_label'] for i in data])
    
    def get_images_with_attribute(self,attribute_name,seed=-1,suffix="",train=True):
        data = self.get_data(seed,suffix,train=train)
        attributes = self.get_attributes()
        attribute_index = attributes.index(attribute_name)
        
        matching_attributes = [self.path_to_image(i['img_path']) 
                               for i in data if i['attribute_label'][attribute_index] == 1]
        return matching_attributes
    
    def get_images_without_attribute(self,attribute_name,one_class=False,seed=-1,suffix="",train=True):
        data = self.get_data(seed,suffix,train=train)
        attributes = self.get_attributes()
        attribute_index = attributes.index(attribute_name)
        
        if one_class:
            all_classes = self.get_class_labels(seed,suffix,train=train)
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


    
    def create_robustness(self):
        flip_probability = .01
        
        for file_name in [self.pkl_path,self.test_pkl_path]:
            input_file = file_name.format('')
            output_file = file_name.format('_image_robustness')
            flip_concept_labels_file(input_file,
                                     output_file,
                                     flip_probability,
                                     self.fix_path,
                                     "_image_robustness")

        self.run_function(self.root_folder_name+"_image_robustness",lambda arr: add_gaussian_noise(arr,0,50))
    
    def create_responsiveness(self):
        flip_probability = 0.5
        for file_name in [self.pkl_path,self.test_pkl_path]:
            input_file = file_name.format('')
            output_file = file_name.format('_image_responsiveness')

            flip_concept_labels_file(input_file,
                                     output_file,
                                     flip_probability,
                                     self.fix_path,
                                     "_image_responsiveness")
        
        self.run_function(self.root_folder_name+"_image_responsiveness",lambda arr: create_junk_image(arr))
    
    def run_function(self,output_folder_name,func):
        all_input_files = glob.glob(self.all_files)

        for input_file in all_input_files:
            corresponding_output = input_file.replace(self.root_folder_name,output_folder_name,1)
            write_new_image_function(input_file,corresponding_output,func)

    
class MNIST_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = dataset_folder+"/colored_mnist{}/images/train.pkl"
        self.test_pkl_path = dataset_folder+"/colored_mnist{}/images/val.pkl"
        self.path_to_image = lambda path: dataset_folder+"/"+path
        self.all_files = dataset_folder+"/colored_mnist/images/*/*.png"
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

    def fix_path(self,path,suffix):
        file_loc = "/".join(path.split("/")[-2:])
        new_path = "colored_mnist{}/images/{}".format(suffix,file_loc)

        return new_path

    
class DSprites_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = dataset_folder+"/dsprites{}/preprocessed/train.pkl"
        self.test_pkl_path = dataset_folder+"/dsprites{}/preprocessed/val.pkl"
        self.path_to_image = lambda path: dataset_folder+"/"+path
        self.all_files = dataset_folder+"/dsprites/images/*.png"
        self.root_folder_name = "dsprites"
        self.experiment_name = "dsprites"
        self.class_names = [str(i) for i in range(100)]
        
    
    def get_attributes(self):
        attributes = [
            "is_white",
            "is_square",
            "is_ellipse",
            "is_heart",
            "is_scale_0.5",
            "is_scale_0.6",
            "is_scale_0.7",
            "is_scale_0.8",
            "is_scale_0.9",
            "is_scale_1",
            "is_orientation_0",
            "is_orientation_90",
            "is_orientation_180",
            "is_orientation_270",
            "is_x_0",
            "is_x_16",
            "is_y_0",
            "is_y_16",
        ] 
        
        return attributes
    def fix_path(self,path,suffix):
        file_loc = "/".join(path.split("/")[-2:])
        new_path = "dsprites{}/{}".format(suffix,file_loc)
        return new_path
    
class Chexpert_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = dataset_folder+"/chexpert{}/preprocessed/train.pkl"
        self.test_pkl_path = dataset_folder+"/chexpert{}/preprocessed/val.pkl"
        self.path_to_image = lambda path: dataset_folder+"/"+path
        self.all_files = dataset_folder+"/chexpert/images/*/study1/*.jpg"
        self.root_folder_name = "chexpert"
        self.experiment_name = "chexpert"
        self.class_names = ["Findings","No Findings"]
        
    
    def get_attributes(self):
        attributes = ["Enlarged Cardiom",
                      "Cardiomegaly",
                      "Lung Lesion",
                      "Lung Opacity",
                      "Edema",
                      "Consolidation",
                      "Pneumonia",
                      "Atelectasis",
                      "Pneumothroax",
                      "Pleural Effusion",
                      "Pleural Other",
                      "Fracture",
                      "Support Devices"]
        
        return attributes
    def fix_path(self,path,suffix):
        file_loc = "/".join(path.split("/")[-4:])
        new_path = "chexpert{}/{}".format(suffix,file_loc)
        return new_path

class CUB_Dataset(Dataset):
    def __init__(self):
        self.pkl_path = dataset_folder+"/CUB{}/preprocessed/train.pkl"
        self.test_pkl_path = dataset_folder+"/CUB{}/preprocessed/val.pkl"
        self.path_to_image = lambda path: dataset_folder+"/"+path
        self.all_files = dataset_folder+"/CUB/images/CUB_200_2011/images/*/*.jpg"
        self.root_folder_name = "CUB"
        self.experiment_name = "cub"
        
        self.class_names = open(dataset_folder+"/CUB/metadata/classes.txt").read().strip().split("\n")
        self.class_names = [i.split(" ")[1] for i in self.class_names]
    
    def get_attributes(self):
        attributes_file = dataset_folder+"/CUB/metadata/attributes.txt"
        lines = open(attributes_file).read().strip().split("\n")
        attributes = [i.split(" ")[1] for i in lines]
        return attributes
    
    def fix_path(self,path,suffix):
        file_loc = "/".join(path.split("/")[-2:])
        new_path = "CUB{}/images/CUB_200_2011/images/{}".format(suffix,file_loc)

        return new_path

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
        
def flip_concept_labels(concept_list,flip_prob,img_path_update,suffix):
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
        data['img_path'] = img_path_update(data['img_path'],suffix)
        
        for i in range(len(data['attribute_label'])):
            if np.random.random() < flip_prob:
                data['attribute_label'][i] = 1-data['attribute_label'][i]
                
    return new_arr

def flip_concept_labels_file(input_file,output_file,flip_probability,img_path_update,suffix):
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
    new_all_concepts = flip_concept_labels(all_concepts,flip_probability,img_path_update,suffix)
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
        folder_location = dataset_folder+"/images/random500_{}".format(folder_num)
        if not os.path.isdir(folder_location):
            os.mkdir(folder_location)
        delete_files_in_directory(folder_location)
                
        random_images = random.sample(image_locations,images_per_folder)
        
        for image in random_images:
            shutil.copy2(image,folder_location)
    
def create_folder_from_attribute(attribute_name,attribute_function,seed=-1,suffix='',num_images=100,train=True):
    """Create a new folder, with the images being all birds with a particular attribute
    
    Arguments:
        attribute_name: String that's an attribute to either the MNIST dataset or the Birds Dataset
        attribute_function: Function that retrieves all matching images, given an attribute
            Either get_mnist_images_by_attribute or get_cub_images_by_attribute
        num_images: How many images to select from the attribute folder, at most
        train: Whether or not to use the training dataset when creating images

    Returns:
        None
        
    Side Effects:
        Creates a new folder, populated with images
            with a particular attribute
    """
    
    if seed > -1:
        random.seed(seed)
    
    image_locations = attribute_function(attribute_name,suffix=suffix,seed=seed,train=train)
    
    folder_location = dataset_folder+"/images/{}_{}_{}".format(attribute_name,seed,suffix)
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
        

def generate_random_orientation(n):
    """Generate n random orientations based on concepts
    
    Arguments: 
        n: The number of orientations we need to generate

    Returns: A list of size n, each with an orientation of size 6"""
    
    orientations = []
    
    choices_by_value = [1,3,6,4,2,2]
    
    while len(orientations) < n:
        generated_orientation = [random.randint(0,i-1) for i in choices_by_value]
        if generated_orientation not in orientations:
            orientations.append(generated_orientation)
                
    return orientations

def image_has_orientation(orient,concepts):
    """Check if an image concept matches the orientation in orient
    
    Arguments: 
        orient: Array representing an orientation
        concepts: An image concepts which matches up with the orientation

    Returns: Boolean True or False"""
    max_range_by_value = [-1,-1,-1,40,32,32]
    choices_by_value = [1,3,6,4,2,2]


    for i in range(len(orient)):
        if max_range_by_value[i] == -1:
            if orient[i] != concepts[i]:
                return False
        else:
            equiv_orient = concepts[i]/max_range_by_value[i]*choices_by_value[i]
            equiv_orient = int(equiv_orient)

            if equiv_orient != orient[i]:
                return False
                
    return True

def get_matching_images(orientation,npz_file):
    """Get the indices of all matching images for a particular orientation
    
    Arguments:
        orientation: List of 6 numbers representing an orientation
        npz_file: Handle for the NPZ file for DSprites
        
    Returns: List of indices for valid images"""
    
    latents = npz_file['latents_classes']
    return [i for i in range(len(latents)) if image_has_orientation(orientation,latents[i])]
    
def write_image(idx,imgs):
    """Write an image, indexed by idx, to the dsprites folder
    
    Arguments:
        idx: Number index into npz_File
        npz_file: Handle for the NPZ file for DSprites
    
    Returns: Nothing
    
    Side Effects: Writes an image to dataset_folder + /dsprites/images/idx.png
    """
    
    bw_arr = imgs[idx]
    rgb_arr = np.repeat(np.expand_dims(bw_arr, axis=2), 3, axis=2) * 255
    img = Image.fromarray(rgb_arr.astype('uint8'), 'RGB')
    img.save(dataset_folder+'/dsprites/images/{}.png'.format(idx))
    
def one_hot_orientation(orientation):
    """Given some orientation, one hot encode it
    
    Arguments: 
        Orientation: List of size 6
    
    Returns: List of size 1+3+6+3+2+2 = 18"""
    
    one_hot = []
    max_range_by_value = [-1,-1,-1,40,32,32]
    choices_by_value = [1,3,6,4,2,2]
    
    for i in range(len(orientation)):
        for j in range(choices_by_value[i]):
            if orientation[i] == j:
                one_hot.append(1)
            else:
                one_hot.append(0)
    
    return one_hot

def binary_to_decimal(binary_list):
    """Convert a list of 0s and 1s to a binary number
    
    Arguments: 
        binary_list: List of 0s and 1s
    
    Returns: Equivalent integer"""
    
    binary_str = ''.join(map(str, binary_list))
    decimal_num = int(binary_str, 2)
    return decimal_num

    

def write_dataset(orientations,npz_file, write_images=False):
    """Write the metadata train.pkl, etc. based on orientations array
    
    Arguments:
        orientations: List of orientations, which is a list of numbers
        npz_file: Handle for the NPZ file for Dsprites
    
    Returns: Nothing
    
    Side Effects: Writes files and metadata pkl
        
    """
    
    imgs = npz_file['imgs']
    images_by_orientation = [get_matching_images(o,npz_file) for o in orientations]
    
    for i in range(len(images_by_orientation)):
        random.shuffle(images_by_orientation[i])
        
    num_train = [0,250]
    num_val = [250,325]
    num_test = [325,400]
    
    num_split = {
        'train': num_train,
        'val': num_val,
        'test': num_test,
    }
    
    files = glob.glob(dataset_folder+'/dsprites/images/*')
    for f in files:
        os.remove(f)

    
    for split in ["train","val","test"]:
        pkl_file_info = []
        pkl_file_loc = dataset_folder+"/dsprites/preprocessed/{}.pkl".format(split)
        
        low, high = num_split[split]
        
        for o, orientation in enumerate(orientations):
            one_hot = one_hot_orientation(orientation)
            for idx in images_by_orientation[o][low:high]:
                d = {
                    'id': idx,
                    'img_path': 'dsprites/images/{}.png'.format(idx),
                    'class_label': binary_to_decimal(one_hot)%100,
                    'attribute_label': one_hot,
                }
                pkl_file_info.append(d)
                
                if write_images:
                    write_image(idx,imgs)
        
        w = open(pkl_file_loc,"wb")
        pickle.dump(pkl_file_info,w)
        w.close()
        
def write_ten_dsprites():
    npz_file = np.load(open(dataset_folder+"/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz","rb"))
    oreintations = generate_random_orientations(10)
    write_dataset(orientations,npz_file, write_images=True)

        