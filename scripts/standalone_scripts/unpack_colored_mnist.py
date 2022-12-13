import numpy as np
from PIL import Image
import pickle 
import random

def write_image_from_numpy_array(arr,location):
    """Turn a mxnx3 numpy array into a png image, stored at location
    
    Arguments:
        arr: Numpy array of size m x n x 3, which contains pixel values 0-255
        location: File location where images should be saved
        
    Returns: Nothing
    
    Side Effects: Saves a new image file at location 
    """
    
    im = Image.fromarray(arr)
    im.save(location)
    
def load_mnist_dictionary():
    """Loads the mnist_10color_jitter_var_0.030.npy from the colored_mnist folder
    
    Arguments: Nothing
    
    Returns: Dictionary with keys: 
        train_label: 60K size list, with each index being a number 0-9
        train_image: 60K x 28 x 28 x 3 numpy array; each index represents an MNIST image
        test_label: 10K size list, with each index being a number 0-9
        test_image: 10K x 28 x 28 x 3 numpy array; each index represents a test MNIST image
        test_gray: 10K x 28 x 28 numpy array; each image corresponds to a black and white
            version of the same image in test image
            Used to see if the model picks up on spurrious correlations 
    """
    
    return np.load(open("colored_mnist/mnist_10color_jitter_var_0.030.npy","rb"),allow_pickle=True,encoding='latin1').item()

def create_dataset(write_images=True):
    """Creates the dataset by writing images to files and the train.pkl dictionary, with information on 
        each data point

    Arguments: Nothing
    
    Returns: Nothing
    
    Side Effects: Writes two sets of files:
        colored_mnist/images/ - Folder containing all 60K images, with 10 folders; one for each class
        train.pkl - Pickled list of dictionaries, with the following keys:
            img_path: Location of the corresponding image, in terms of colored_mnist/images/0/0.png
            class_label: Integer 0-9 representing what nuimber it is
    """
    
    mnist_dictionary = load_mnist_dictionary()
    train_dictionary = []
    valid_dictionary = []
    
    for split in ['train','valid']:
        image_key = '{}_image'.format(split)
        label_key = '{}_label'.format(split)
        for i in range(len(mnist_dictionary[image_key])):
            image = mnist_dictionary[image_key][i]
            label = mnist_dictionary[label_key][i]

            image_number = i 
            if split == 'valid':
                image_number += len(mnist_dictionary['train_image'])

            storage_location = "colored_mnist/images/{}/{}.png".format(label,image_number)

            info_dictionary = {'img_path': storage_location, 
                                'class_label': label, 
                                'id': image_number}
            attribute_label = []

            
            for j in range(10):
                if label == j:
                    value = 1
                else:
                    value = 0

                info_dictionary['{}_color'.format(j)] = value
                info_dictionary['{}_number'.format(j)] = value
                
                attribute_label.append(value)
                attribute_label.append(value)
            
            info_dictionary['spurious'] = random.randint(0,1)
            attribute_label.append(info_dictionary['spurious'])
            
            info_dictionary['attribute_label'] = attribute_label
            
            train_dictionary.append(info_dictionary)
            
            if write_images:
                write_image_from_numpy_array(image,storage_location)
        
    pickle.dump(train_dictionary,open("colored_mnist/images/train.pkl","wb"))
    
if __name__ == "__main__":
    create_dataset(write_image = False)