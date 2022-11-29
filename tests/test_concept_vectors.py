from src.concept_vectors import *
from src.download_data import *
import os
import numpy as np

def test_cem():
    assert len(load_cem_vectors("xor",0))>0
    assert len(load_cem_vectors("xor",1))>0
    
    xor_concepts = {}
    for i in [0,1]:
        for j in ["active","inactive"]:
            xor_concepts["{}_{}".format(i,j)] = np.load("./results/cem_concepts/xor_concept_{}_{}.npy".format(i,j))

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
    
    assert create_tcav_vectors(concepts,target,model_name,bottlenecks,num_random_exp,alphas=alphas) == None
    
    assert os.path.exists("./results/cavs/soccer-random500_0-mixed4c-linear-0.1.pkl")
    assert os.path.exists("./results/cavs/alga-random500_0-mixed4c-linear-0.1.pkl")
    
    soccer_tcav, soccer_metadata = load_tcav_vectors("soccer",bottlenecks)
    assert soccer_tcav.shape[0] == 2
    assert soccer_tcav.shape[1] > 100

    alga_tcav, alga_metadata = load_tcav_vectors("alga",bottlenecks)
    assert alga_tcav.shape[0] == 2
    assert alga_tcav.shape[1] > 100
    
if __name__ == "__main__":
    test_tcav()
    test_cem()