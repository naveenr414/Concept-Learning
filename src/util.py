import numpy as np
from scipy.spatial import distance_matrix
from collections import Counter

def find_average_distances(vector_list,class_list):
    """Given a list of vectors, each a member of a certain class, find the average distance between pairs of classes
    
    Arguments:
        vector_list: n by k matrix of vectors
        class_list: int list of length n, each index representing membership in a certain class
            number 0...r-1
        
    Returns:
        An rxr matrix with average distances between each class, where r=# of classes
    """
    
    distances = distance_matrix(vector_list,vector_list)
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
            