from src.concept_vectors import *
from src.download_data import *
import os
import numpy as np
from src.dataset import *

def test_label_cub():
    dataset = CUB_Dataset()
    all_attributes = dataset.get_attributes()

    assert len(create_vector_from_label(all_attributes[0],dataset)) > 0
    assert len(create_vector_from_label(all_attributes[1],dataset)) > 0

def test_label_mnist():
    dataset = MNIST_Dataset()
    all_attributes = dataset.get_attributes()

    assert len(create_vector_from_label(all_attributes[0],dataset)) > 0
    assert len(create_vector_from_label(all_attributes[1],dataset)) > 0
    
def test_tcav_cub():
    dataset = CUB_Dataset()
    
    all_attributes = dataset.get_attributes()
    
    image_locations = dataset.get_images_with_attribute(all_attributes[0])
    assert len(image_locations) > 0
    assert os.path.exists(image_locations[0])    
    
    assert len(dataset.get_images_with_attribute(all_attributes[10])) > 0
    
    assert create_tcav_dataset(all_attributes[0],dataset,2,seed=42) == None
    assert os.path.exists("./results/cavs/cub/42/{}-random500_0-mixed4c-linear-0.1.pkl".format(all_attributes[0]))
    
def test_cem():
    assert len(load_cem_vectors("xor",0))>0
    assert len(load_cem_vectors("xor",1))>0
    
    xor_concepts = {}
    for i in [0,1]:
        for j in ["active","inactive"]:
            xor_concepts["{}_{}".format(i,j)] = np.load("./results/cem_concepts/xor/42/xor_concept_{}_{}.npy".format(i,j))

    assert len(xor_concepts['0_active']) + len(xor_concepts['0_inactive']) ==  \
        len(xor_concepts['1_active']) + len(xor_concepts['1_inactive']) 

def test_tcav():
    concepts = ["soccer","alga"]
    target = "otter"
    model_name = "GoogleNet"
    bottlenecks = ["mixed4c"]
    num_random_exp = 2
    alphas = [0.1]
    
    assert download_imagenet(concepts,10) == None
    assert download_imagenet([target],10) == None
    if not os.path.exists("./dataset/random500_0") or len(os.listdir("./dataset/images/random500_0")) == 0:
        assert download_random_imagenet_classes(1,10) == None
    
    assert create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,alphas=alphas,experiment_name="unfiled",seed=42) == None
    
    assert os.path.exists("./results/cavs/unfiled/42/soccer-random500_0-mixed4c-linear-0.1.pkl")
    assert os.path.exists("./results/cavs/unfiled/42/alga-random500_0-mixed4c-linear-0.1.pkl")
    
    soccer_tcav, soccer_metadata = load_tcav_vectors("soccer",bottlenecks,experiment_name="unfiled",seed=42)
    assert soccer_tcav.shape[0] == 2
    assert soccer_tcav.shape[1] > 100

    alga_tcav, alga_metadata = load_tcav_vectors("alga",bottlenecks,experiment_name="unfiled",seed=42)
    assert alga_tcav.shape[0] == 2
    assert alga_tcav.shape[1] > 100
    
if __name__ == "__main__":
    test_cem()
    test_tcav_cub()
    test_label_cub()
    test_label_mnist()
    test_tcav()
