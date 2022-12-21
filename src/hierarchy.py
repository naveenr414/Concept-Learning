from scipy.cluster.hierarchy import dendrogram, linkage, ward
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

def create_ward_hierarchy(data,metric='euclidean'):
    """Use the sklearn ward function to construct a hierarchal cluster
    
    Arguments:
        data: Numpy array of data/array of numpy vectors

    Returns: 
        numpy array with information on how to construct a dendogram
    """
    
    dist = pdist(data,metric=metric)
    return ward(dist)

def create_linkage_hierarchy(data,method='single',metric='euclidean'):
    """Use the sklearn linkage function to construct a hierarchal cluster
    
    Arguments:
        data: Numpy array of data/array of numpy vectors
        method: What type of linkage, such as 'single', 'complete', or 'average'
        mertic: What metric to use, such as 'euclidean', 'chebyshev', 'cosine', etc.

    Returns:
        numpy array with information on how to construct a dendogram
        This can be fed into the dendrogram method to plot it
    """
    
    return linkage(data,method=method,metric=metric)

def create_hierarchy_thresholding(data,metric='euclidean'):
    """Use a thresholding algorithm with a minimum spanning tree to construct a hierarcha cluster
    
    Arguments:
        data: Numpy array of data/array of numpy vectors
        metric: What metric to use, such as 'euclidean', 'chebyshev', 'cosine', etc. 
        
    Returns:
        numpy array with information on how to construct a dendrogram
        This can be fed into the dendrogram method to plot it

    """
    
    distance_matrix = cdist(data,data,metric=metric)
    tree = minimum_spanning_tree(distance_matrix)
    tree = tree.toarray().astype(int)
    
    node_combinations = []
    for i in range(len(tree)):
        for j in range(len(tree[i])):
            if tree[i][j]>0:
                node_combinations.append((i,j,tree[i][j]))

    node_combinations = sorted(node_combinations,key=lambda k: k[2])
    
    # Create a matrix in the dendrogram format (https://stackoverflow.com/questions/9838861/scipy-linkage-format)
    return_matrix = np.zeros((len(data)-1,4))
    new_clusters = [[i] for i in range(len(data))]
    
    current_num = 0
    
    for u,v,dist in node_combinations:
        # Combine nodes u, v
        u_cluster = max([i for i in range(len(new_clusters)) if u in new_clusters[i]])
        v_cluster = max([i for i in range(len(new_clusters)) if v in new_clusters[i]])
            
        cluster_size = len(new_clusters[u_cluster]) + len(new_clusters[v_cluster])

        new_cluster = new_clusters[u_cluster] + new_clusters[v_cluster]
        new_cluster = list(set(new_cluster))
        new_clusters.append(new_cluster)
        
        return_matrix[current_num] = np.array([u_cluster,v_cluster,dist,cluster_size])
        current_num += 1
        
    return return_matrix