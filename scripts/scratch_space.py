import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tcav
from src.download_data import *
import pickle
from src.dataset import *
from src.concept_vectors import *

cav_dir = './results/cavs'
activation_dir = './results/activations'
working_dir = './results/tmp'
image_dir = "./dataset/images"

attribute_list = """has_bill_shape::dagger
has_bill_shape::hooked_seabird
has_bill_shape::all-purpose
has_bill_shape::cone
has_leg_color::grey
has_leg_color::black
has_leg_color::buff
has_bill_color::grey
has_bill_color::black
has_bill_color::buff""".strip().split("\n")

for attribute in attribute_list:
    print("On attribute {}".format(attribute))
    create_tcav_cub(attribute,20)