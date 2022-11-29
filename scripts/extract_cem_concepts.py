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

n_concepts = 2
n_tasks = 2
n_features = 2
experiment_name = "xor"
num_gpus = 0

def c_extractor_arch(output_dim):
    """A feedforward architecture used before concept extraction 
    
    Returns: 
        A Torch Architecture which consists of a series of layers
    """
    
    return torch.nn.Sequential(*[
        torch.nn.Linear(n_features, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, output_dim),
    ])

def generate_data_loaders():
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

train_dl, valid_dl = generate_data_loaders()

cem_model = ConceptEmbeddingModel(
  n_concepts=n_concepts, # Number of training-time concepts
  n_tasks=n_tasks, # Number of output labels
  emb_size=16,
  concept_loss_weight=0.1,
  learning_rate=1e-3,
  optimizer="adam",
  c_extractor_arch=c_extractor_arch, # Replace this appropriately
  training_intervention_prob=0.25, # RandInt probability
  experiment_name=experiment_name
)

trainer = pl.Trainer(
    gpus=num_gpus,
    max_epochs=2,
    check_val_every_n_epoch=1,
)

trainer.fit(cem_model, train_dl, valid_dl)
