import torch
from experiments import intervention_utils
import os
from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
from torchvision.models import resnet50, resnet34
from cem.models.cem import ConceptEmbeddingModel
import pytorch_lightning as pl
import numpy as np
import cem.train.training as cem_train
import re
import random
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from vae_model import *
    

def intervention_results(related_concepts, test_range=range(0,len(intervention_utils.CUB_CONCEPT_GROUP_MAP),4),trials=3):
    CUB_DIR = 'CUB/'
    BASE_DIR = os.path.join(CUB_DIR, 'images/CUB_200_2011/images')
    num_workers=8
    n_tasks = 200
    n_concepts = 112
    gpu = 1 if torch.cuda.is_available() else 0
    sample_train = 0.1
    concept_group_map = intervention_utils.CUB_CONCEPT_GROUP_MAP
    num_epochs = 100
    seed = 42

    train_data_path = os.path.join(CUB_DIR, 'preprocessed/train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

    config = joblib.load("results/ConceptEmbeddingModelNew_resnet34_fold_1_experiment_config.joblib")
    if config['weight_loss']:
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None
    
    selected_concepts = np.arange(n_concepts)
    def subsample_transform(sample):
        if isinstance(sample, list):
            sample = np.array(sample)
        return sample[selected_concepts]
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='.',
        num_workers=config['num_workers'],
        concept_transform=subsample_transform,
        path_transform=lambda path: path.replace("CUB//",""),
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='.',
        num_workers=config['num_workers'],
        concept_transform=subsample_transform,
        path_transform=lambda path: path.replace("CUB//",""))
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='.',
        num_workers=config['num_workers'],
        concept_transform=subsample_transform,
        path_transform=lambda path: path.replace("CUB//","")
    )
    
    
    if sample_train < 1.0:
        train_dataset = train_dl.dataset
        train_size = round(len(train_dataset)*sample_train)
        train_subset = random.sample(range(0,len(train_dataset)),train_size)
        train_dataset = torch.utils.data.Subset(train_dataset, train_subset)
        sample_train_dl = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True, 
                                               drop_last=True, num_workers=num_workers)

    trainer = pl.Trainer(
            gpus=gpu,
        )
    
    
    results_by_num_groups = {}
    for num_groups_intervened in test_range:
        results_by_num_groups[num_groups_intervened] = []
    
    config["related_concepts"] = related_concepts
    
    for trial in range(trials):
        pl.seed_everything(42+trial)
        for num_groups_intervened in test_range:
            intervention_idxs = intervention_utils.random_int_policy(
                        num_groups_intervened=num_groups_intervened,
                        concept_group_map=concept_group_map,
                        config=config,
                    )

            model = intervention_utils.load_trained_model(
                        config=config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        result_dir="results/",
                        split=0,
                        imbalance=imbalance,
                        intervention_idxs=intervention_idxs,
                        train_dl=train_dl,
                        sequential=False,
                        independent=False,
                    )

            [test_results] = trainer.test(model, val_dl, verbose=False,)
            results_by_num_groups[num_groups_intervened].append(test_results)
    return results_by_num_groups

def get_random_related_concepts():
    random_related = {}
    num_random_per = 10
    identity_function = lambda s: s
    
    n_concepts = 112

    for i in range(n_concepts):
        other_concepts = [j for j in range(n_concepts) if j!= i]
        our_concepts = random.sample(other_concepts,num_random_per)
        confidences = [random.random() for j in our_concepts]

        random_related[i] = {}

        for j in range(len(our_concepts)):
            random_related[i][our_concepts[j]] = (identity_function,confidences[j])
            
    return random_related

def find_closest_vectors(matrix):
    num_vectors = matrix.shape[0]
    sim_matrix = cosine_similarity(matrix)

    # Set diagonal to -inf to exclude self-similarity
    np.fill_diagonal(sim_matrix, -np.inf)

    closest_indices = []
    for i in range(num_vectors):
        cosine_similarities = sim_matrix[i]
        distances = [(j, abs(sim),sim) for j, sim in enumerate(cosine_similarities)]
        # Sort by distance in descending order
        distances.sort(key=lambda x: x[1], reverse=True)
        # Get the indices of the 3 closest vectors (excluding itself)
        closest_indices.append([(j, sim) for j, d, sim in distances if j != i][:10])

    return closest_indices

def get_related_concepts_file(file_name):
    vectors = np.load(open(file_name,"rb"))
    closest_vectors = find_closest_vectors(vectors)
    n_concepts = 112
    
    related_vectors = {}
    identity_function = lambda s: s
    opposite_function = lambda s: 1-s

    for i in range(n_concepts):
        related_vectors[i] = {}

        for index,similarity in closest_vectors[i]:
            confidence = similarity
            if similarity > 0:
                related_vectors[i][index] = (identity_function,confidence)
            else:
                related_vectors[i][index] = (opposite_function,confidence)
                
    return related_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concept vectors based on ImageNet Classes')
    parser.add_argument('--algorithm',type=str,
                        help='Which algorithm to use to generate related concept vectors')
    args = parser.parse_args()
    
    output_file = "results/intervention_{}.txt".format(args.algorithm)
    if args.algorithm == 'normal':
        related_concepts = None
    elif args.algorithm == 'random':
        related_concepts = get_random_related_concepts()
    elif args.algorithm == 'concept2vec':
        related_concepts = get_related_concepts_file('concept_vectors/concept2vec.npy')
    elif args.algorithm == 'labels':
        related_concepts = get_related_concepts_file('concept_vectors/labels.npy')

    results = intervention_results(related_concepts)
    w = open(output_file,"w")
    
    for num_groups_intervened in results:
        for trial_id,trial in enumerate(results[num_groups_intervened]):
            w.write("{},{},{}\n".format(num_groups_intervened,trial_id,trial['test_y_accuracy']))
    w.close()

    
