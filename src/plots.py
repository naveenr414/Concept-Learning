import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram
from PIL import Image

color_palette = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628']
extra_large_color_palette = ['#984ea3','#999999', '#e41a1c', '#dede00','#994f00'] 
extra_large_color_palette = color_palette + extra_large_color_palette
scatter_shapes = ["o","v","D","s","*"]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_palette)
plt.style.use('ggplot')

def filter_data_labels(data,labels,filter_labels):
    """
    Filters a dataset + labels based on a filter function
    Each label is fed into the filter_labels function; 
        those which return True are kept
        
    Arguments: 
        data: Numpy matrix of data
        labels: String label for each data point
        filter_labels: Boolean function that asserts if 
            each data point should be kept

    Returns:
        data: Numpy matrix of data
        labels: String label for each data point
    """
    useful_labels = [index for index in range(len(labels)) if filter_labels(labels[index])]
    useful_labels_set = set(useful_labels)
    data = data[useful_labels]
    labels = [label for i,label in enumerate(labels) if i in useful_labels_set] 
    return data, labels

def scatter_labels(data,labels=[],filter_labels=None):
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
    
    if filter_labels:
        data, labels = filter_data_labels(data,labels,filter_labels)

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

def plot_tsne(data,labels=[],filter_labels=None): 
    """
    Embeds high-dimensional data into 2D dimensions using TSNE algorithm, then plots it
    
    Arguments:
        data: Numpy matrix of data
        labels (optional): Label for each data point
    
    Returns:
        None
        
    Side Effects:
        Plots scatter plot using data
    """
        
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(data)
    scatter_labels(X_embedded,labels,filter_labels)
    
def plot_pca(data,labels=[],filter_labels=None): 
    """
    Embeds high-dimensional data into 2D dimensions using PCA algorithm, then plots it
    
    Arguments:
        data: Numpy matrix of data
        labels (optional): Label for each data point
        filter_labels (optional): Function that determines which datapoints
            should be kept
    
    Returns:
        None
        
    Side Effects:
        Plots scatter plot using data
    """
    
    X_embedded = PCA(n_components=2).fit_transform(data)
    scatter_labels(X_embedded,labels,filter_labels)
    
def generate_labels_from_activation(concept_list,activation):
    """From a list of concepts, generate the labels when performing scatter plots
        using these concepts, so that each concept appears #concept vectors times

    Arguments:
        concept_list: String list of concepts, which will be analyzed
        activation: Dictionary, which says how many concept vectors are present for each
            concept (in the form of a numpy array containing all concepts)

    Returns:
        String list, with concepts repeated n times, where n is the number of concept vectors
        
    Example: generate_labels_from_activation(['hello','goodbye'],{'hello': 2x100 list, 'goodbye': 3x100 list})
        Returns: ['hello','hello','goodbye','goodbye','goodbye']
    """
    
    return_list = []
    for i in concept_list:
        return_list += [i for j in range(len(activation[i]))]
    
    return return_list

def plot_arrows(origins,arrows):
    """Plot a series of arrows, originating at each origin (an nx2 matrix) and 
        ending at each endpoint in arrows (another nx2 matrix) 
        
    Arguments:
        origins: A numpy array of where each arrow starts from
        arrows: A numpy array of where each arrow ends
    """
    
    plt.quiver(origins[:,0],origins[:,1], arrows[:,0], arrows[:,1], color=color_palette, scale=21)
    
def plot_dendogram(tree_info,labels):
    """Plot a tree diagram for a hierarchal cluster
    
    Arguments: 
        tree_info: Numpy array arising from a call to the sklearn hierarchal clustering method
        labels: Label for each vector whihc is clustered

    Returns: Nothing
    
    Side Effects: Plots Tree Diagram/dendogram
    """
    
    labels = [i[:20] for i in labels]
    plt.figure(figsize =(15,7))
    dendrogram(tree_info,labels=labels,orientation='right',distance_sort='descending') 
    
def plot_latent_space(vae, n=30, figsize=15,channels=3,coords=[0,1],latent_size=2):
    """Plot the results for a VAE in a grid with images for different latent representations
    
    Arguments:
        vae: A VAE model from Keras
        n: Size of the grid; the 2D plane from -1 to 1 will be split into nxn data points
        figsize: Number of pixels per image (each dimension)
        channels: Number of channels in an image; 3 for RGB

    Returns: Nothing
    
    Side Effects: Plots a grid of images
    """
    
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n,channels))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    grid_coord_to_flat = {}
    
    z_sample = []
    count = 0
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            grid_coord_to_flat[(i,j)] = count
            
            full_latent = np.zeros(latent_size)
            full_latent[coords[0]] = xi
            full_latent[coords[1]] = yi
                        
            z_sample.append(list(full_latent))
            
            count += 1
            
    z_sample = np.array(z_sample)
    print(z_sample.shape)
            
    x_decoded = vae.decoder.predict(z_sample)
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            digit = x_decoded[grid_coord_to_flat[(i,j)]].reshape(digit_size, digit_size,channels)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()

def plot_image(file_path,resize=False):
    """Given a file_path, plot the image
    
    Arguments:
        file_path: String, representing the path to an image
        
    Returns: Nothing
    
    Side Effects: Plots the image
    """
    image = Image.open(file_path)
    
    if resize:
        image = image.resize((64,64))
    plt.imshow(image)
    plt.grid(False)
    plt.show()
