import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

color_palette = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628']
extra_large_color_palette = ['#984ea3','#999999', '#e41a1c', '#dede00','#994f00'] 
extra_large_color_palette = color_palette + extra_large_color_palette
scatter_shapes = ["o","v","D","s","*"]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_palette)
plt.style.use('ggplot')

def scatter_labels(data,labels=[]):
    """
    Helper function to scatter points with labels for each point
    
    Arguments:
        data: List of numpy arrays (or numpy matrix) of data
        labels (optional): Labels for each data point; if it's empty, 
            then a regular scatter plot will be plotted
        
    Returns:
        None
    
    Side Effects:
        Plots scatter plot with labels
    """
    
    if labels == []:
        plt.scatter(data)
        return
    
    data_by_label = defaultdict(lambda: [])
    for i in range(len(labels)):
        data_by_label[labels[i]].append(data[i])
       
    for i,label in enumerate(data_by_label):
        data_by_label[label] = np.array(data_by_label[label])
        color = color_palette[i%len(color_palette)]
        shape = scatter_shapes[i//len(color_palette)]
        plt.scatter(data_by_label[label][:,0],data_by_label[label][:,1],label=label,
                    c=color,marker=shape)
    plt.legend()
        

def plot_tsne(data,labels=[]): 
    """
    Embeds high-dimensional data into 2D dimensions using TSNE algorithm, then plots it
    
    Arguments:
        data: Numpy matrix of data
        colors (optional): Color of each data point in the scatter plot; 
            used when data comes from separate classes or categories
        labels (optional): Labe for each data point
    
    Returns:
        None
        
    Side Effects:
        Plots scatter plot using data
    """
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(data)
    scatter_labels(X_embedded,labels)
    
def plot_pca(data,labels=[]): 
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
    scatter_labels(X_embedded,labels)