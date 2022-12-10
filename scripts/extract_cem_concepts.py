import pytorch_lightning as pl
from cem.models.cem import ConceptEmbeddingModel
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import resnet50
from experiments.synthetic_datasets_experiments import generate_xor_data
from torch.utils.data import TensorDataset, DataLoader
from cem.data.CUB200.cub_loader import load_data
import numpy as np

experiment_name = "cub"
num_gpus = 1
num_epochs = 1
validation_epochs = 1
num_workers = 16

def retrieve_selected_concepts():
    """The CBM and CEM papers use a subset of the main birds concept
        In particular, they use 112/312 concepts
        We retrieve the indexes of these concepts; which we precompute and store elsewhere
        
        Arguments: None
        
        Returns: List of 0-indexed indicies, which represent attributes that are used
    """
    
    attributes_file = "../../main_code/dataset/CUB/metadata/attributes.txt"
    
    f = open(attributes_file).read().strip().split("\n")
    indexes = [int(i.split(" ")[0]) for i in f]
    
    # Make indexes 0-indexed
    indexes = [i-1 for i in indexes]
    return indexes

selected_concepts = retrieve_selected_concepts()

def c_extractor_arch(output_dim):
    """A feedforward architecture used before concept extraction 
    
    Returns: 
        A Torch Architecture which consists of a series of layers
    """
    
    return torch.nn.Sequential(*[
        torch.nn.Linear(2, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, output_dim),
    ])

def subsample_transform(sample):
    if isinstance(sample, list):
        sample = np.array(sample)
    return sample[selected_concepts]

def generate_data_loaders_xor():
    """Generate the train and validation dataloaders that are fed into the model
    Modify this as the data we used changes
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader consisting of data (x), output (y), and concepts (c)
        valid_dl: A PyTorch dataloader with the data, output, and concepts
    """

    x,c,y = generate_xor_data(100)
    x = x.type(torch.FloatTensor)
    y = y.type(torch.LongTensor)
    c = c.type(torch.FloatTensor)

    train_dl = DataLoader(TensorDataset(x,y,c))
    valid_dl = DataLoader(TensorDataset(x,y,c))
    
    return train_dl, valid_dl

def generate_data_loaders_cub():
    """Generate the train and validation dataloaders for the birds dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    cub_location = '../ConceptBottleneck/CUB/data'
    train_data_path = cub_location+'/class_attr_data_10/train.pkl'
    valid_data_path = cub_location+'/class_attr_data_10/val.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/CUB_200_2011/images/',
        resampling=False,
        root_dir=cub_location+'/CUB_200_2011',
        num_workers=num_workers,
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/CUB_200_2011/images/',
        resampling=False,
        root_dir=cub_location+'/CUB_200_2011',
        num_workers=num_workers,
    )
    
    return train_dl, valid_dl

if experiment_name == "xor":
    train_dl, valid_dl = generate_data_loaders_xor()
    n_concepts = 2
    n_tasks = 2

elif experiment_name == "cub":
    train_dl, valid_dl = generate_data_loaders_cub()
    n_concepts = 112
    n_tasks = 200

    
if experiment_name == "xor":
    extractor_arch = c_extractor_arch
elif experiment_name == "cub":
    extractor_arch = resnet50

cem_model = ConceptEmbeddingModel(
  n_concepts=n_concepts, # Number of training-time concepts
  n_tasks=n_tasks, # Number of output labels
  emb_size=16,
  concept_loss_weight=0.1,
  learning_rate=0.01,
  optimizer="adam",
  c_extractor_arch=extractor_arch, # Replace this appropriately
  training_intervention_prob=0.25, # RandInt probability
  experiment_name=experiment_name
)

trainer = pl.Trainer(
    gpus=num_gpus,
    max_epochs=num_epochs,
    check_val_every_n_epoch=validation_epochs,
)

trainer.fit(cem_model, train_dl, valid_dl)
cem_model.write_concepts()