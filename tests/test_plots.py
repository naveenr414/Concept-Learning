from src.plots import *
import numpy as np

def test_dimensionality_reduction():
    """
    Test out the PCA and TSNE plotting algorithms
    """
    assert plot_tsne(np.array([[0,1],[2,3],[4,5]])) == None
    assert plot_tsne(np.array([[0,1],[2,3],[4,5]]),['r','g','b']) == None   
    
    assert plot_pca(np.array([[0,1],[2,3],[4,5]])) == None
    assert plot_pca(np.array([[0,1],[2,3],[4,5]]),['r','g','b']) == None   

if __name__ == 'main':
    test_dimensionality_reduction()
