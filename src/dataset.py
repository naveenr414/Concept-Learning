import numpy as np
import pickle

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