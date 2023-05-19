# Concept Hierarchies for Concept Learning Methods

![Concept Hierarchies](figures/pull_figure.png)

This repository contains the implementation for the thesis "Concept Hierarchies for Concept Learning Methods", for the Masters in Philosophy course at Cambridge University.

This work was done by [Naveen Raman](https://naveenraman.com/)

#### TL;DR
We construct concept hierarchies, a way to capture concept-concept relationships in concept models. These hierarchies provide a qualitative visual summary of a model, while being stable across iterations and robust to noise. Additionally, concept hierarchies assist with downstream applications include concept intervention and classification. 

We provide code here to perform the following operations: 
1. Construct concept hierarchies and concept vectors
2. Evaluate concept hierarchies
3. Employ concept hierarchies for concept intervention and classification

We provide the bulk of the code in this repository. However, to run concept interventions, we provide information in the scripts/cem_scripts folder. 

## Installation and Datasets
### Installation
To install dependencies, run the following
```
$ conda env create --file environment.yaml
```

### Datasets
We use four datasets for the project: CUB, CheXpert, DSprites, and colored MNIST:

1. <b>Colored MNIST:</b> We download the Colored MNIST dataset from <a href="https://drive.google.com/u/0/uc?id=1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu&export=download">here</a>, and use the ```mnist_10color_jitter_var_0.030.npy``` variant.
2. <b>CUB:</b> We download the CUB dataset from <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">here</a>. 
3. <b>DSprites:</b> We develop DSprites from .npz files in the DSprites directory; to create the dataset, run the following: 
```python

from src.dataset import write_ten_dsprites
write_ten_dsprites()
```
4. <b>CheXpert:</b> We use the small variant of CheXpert from <a href="https://www.kaggle.com/datasets/ashery/chexpert">here</a>


Install each to the dataset/images folder; for examples on what image paths should look like, use the preprocessed/train.py file. 

## Constructing Concept Hierarchies
We give instructions on how to develop each of the following concept vectors: Label,Shapley,Concept2Vec, and detail CEM later
1. <b>Label:</b> To construct label vectors, simply run the following function
```python
load_label_vectors_simple(attribute,dataset,suffix,seed=-1)
```
where suffix is either "", "_image_robustness", or "_image_responsiveness". Dataset is an object from the Dataset class in `dataset.py`, and attribute is a string representing a concept. 
2. <b>Shapley:</b> 
3. <b>Concept2Vec:</b> 

## Evaluating Concept Hierarchies
After creating concept vectors, we evaluate them in the `scripts/Evaluate Hierarchies.ipynb`
<b> Discuss everything this requires </b>

## Concept Intervention and Training CEM Vectors
<b> Insert how to set up CEM </b> 
### Training CEM Vectors
To train CEM vectors, we run the following: 
```bash

$ python experiments/extract_cem_concepts.py --experiment_name $experiment_name --num_gpus $num_gpus --num_epochs 50 --validation_epochs 25 --seed $seed --concept_pair_loss_weight 0
```
In this, `experiment_name` is one of mnist, cub, dsprites, or chexpert, `num_gpus` is the number of GPUs available, and `seed` is the random seed. The resulting vectors are stored at `cem_concepts` folder. 

### Concept Intervention
We perform concept intervention using hierarchies in the `notebooks/CEM Intervention Experiments.ipynb` file. This requires a previously trained CEM file <b> Insert details on how to store a CEM file </b>. Additionally, this requires concept vectors in the `concept_vectors` folder, which we generate by <b>Insert details on how to generate this file</b>. 

## Classification and other Applications
We leverage concept hierarchies to fix concept values predicted by encoder models. 

<b> Insert how to generate the logits </b> 

If the resulting logits are stored at `results/logits/CUB/train_c.npy`, etc., then we run the `scripts/Downstream Experiments.ipynb` to fix concept predictions, and evaluate these predictions over a KNN model. These new logit predictions are stored at `results/logits/CUB/train_c_fixed.npy` (etc.), and are used to train a decoder. We train this decoder using an MLP model in `scripts/Hierarchical CBM.ipynb`. 