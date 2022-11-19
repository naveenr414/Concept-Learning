import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(data, colors=[]): 
    """
    Embeds high-dimensional data into 2D dimensions using TSNE algorithm, then plots it
    
    Arguments:
        data: Numpy matrix of data
        colors (optional): Color of each data point in the scatter plot; 
            used when data comes from separate classes or categories
    
    Returns:
        None
        
    Side Effects:
        Plots scatter plot using data
    """
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(data)

    if len(colors)>0:
        plt.scatter(X_embedded[:,0],X_embedded[:,1],c=colors)
    else:
        plt.scatter(X_embedded[:,0],X_embedded[:,1])
    
def plot_pca(data,colors=[]): 
    """
    Embeds high-dimensional data into 2D dimensions using PCA algorithm, then plots it
    
    Arguments:
        data: Numpy matrix of data
        colors (optional): Color of each data point in the scatter plot; 
            used when data comes from separate classes or categories
    
    Returns:
        None
        
    Side Effects:
        Plots scatter plot using data
    """
    X_embedded = PCA(n_components=2).fit_transform(data)
        
    if len(colors)>0:
        plt.scatter(X_embedded[:,0],X_embedded[:,1],c=colors)
    else:
        plt.scatter(X_embedded[:,0],X_embedded[:,1])