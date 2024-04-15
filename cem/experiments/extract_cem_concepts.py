import pytorch_lightning as pl
from cem.models.cem import ConceptEmbeddingModel
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, vgg16, resnet34
from experiments.synthetic_datasets_experiments import generate_xor_data
from torch.utils.data import TensorDataset, DataLoader
from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
import numpy as np
import argparse
import random
import cem.train.training as cem_train
import pytorch_lightning as pl
import cem.train.training as training
import json
import os

def save_vgg16_model(suffix):
    model = vgg16(pretrained=True)
    
    for name, param in model.named_parameters():
        if suffix == "":
            random_nums = np.ones(param.shape)
        elif suffix == "_model_robustness":
            random_nums = np.random.uniform(low=0.95,high=1.05,size=param.shape)
        elif suffix == "_model_responsiveness":
            random_nums = np.random.uniform(low=-1,high=1,size=param.shape)
        param.data *= torch.Tensor(random_nums)
    
    torch.save(model.state_dict(),"vgg16{}.pt".format(suffix))

def save_vgg16_model(suffix):
    model = vgg16(pretrained=True)
    
    for name, param in model.named_parameters():
        if suffix == "":
            random_nums = np.ones(param.shape)
        elif suffix == "_model_robustness":
            random_nums = np.random.uniform(low=0.95,high=1.05,size=param.shape)
        elif suffix == "_model_responsiveness":
            random_nums = np.random.uniform(low=-1,high=1,size=param.shape)
        param.data *= torch.Tensor(random_nums)
    
    torch.save(model.state_dict(),"vgg16{}.pt".format(suffix))
    
def save_resnet_model(suffix):
    model = resnet50(pretrained=True)
    
    for name, param in model.named_parameters():
        if suffix == "":
            random_nums = np.ones(param.shape)
        elif suffix == "_model_robustness":
            random_nums = np.random.uniform(low=0.95,high=1.05,size=param.shape)
        elif suffix == "_model_responsiveness":
            random_nums = np.random.uniform(low=-1,high=1,size=param.shape)
        param.data *= torch.Tensor(random_nums)
    
    torch.save(model.state_dict(),"resnet{}.pt".format(suffix))

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

def generate_data_loaders_cub(suffix):
    """Generate the train and validation dataloaders for the birds dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    if suffix in ["_model_robustness","_model_responsiveness"]:
        suffix = ""

    cub_location = '../../../datasets/CUB{}'.format(suffix)
    train_data_path = cub_location+'/preprocessed/train.pkl'
    valid_data_path = cub_location+'/preprocessed/val.pkl'
    test_data_path = cub_location+'/preprocessed/test.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images/CUB_200_2011/images/',
        resampling=False,
        root_dir=cub_location+'/images/CUB_200_2011',
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images/CUB_200_2011/images/',
        resampling=False,
        root_dir=cub_location+'/images/CUB_200_2011',
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images/CUB_200_2011/images/',
        resampling=False,
        root_dir=cub_location+'/images/CUB_200_2011',
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    return train_dl, valid_dl, test_dl

def generate_data_loaders_mnist(suffix):
    """Generate the train and validation dataloaders for the MNIST dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    
    if suffix in ["_model_robustness","_model_responsiveness"]:
        suffix = ""
    
    mnist_location = '../../../datasets/colored_mnist{}'.format(suffix)
    train_data_path = mnist_location+'/preprocessed/train.pkl'
    valid_data_path = mnist_location+'/preprocessed/val.pkl'
    test_data_path = mnist_location+'/preprocessed/test.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    return train_dl, valid_dl, test_dl

def generate_data_loaders_mnist_correlated(suffix,correlation_rate):
    """Generate the train and validation dataloaders for the MNIST dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    
    if suffix in ["_model_robustness","_model_responsiveness"]:
        suffix = ""
    
    mnist_location = '../../../datasets/colored_mnist{}'.format(suffix)
    train_data_path = mnist_location+'/preprocessed/train.pkl'
    valid_data_path = mnist_location+'/preprocessed/val.pkl'
    test_data_path = mnist_location+'/preprocessed/test.pkl'
    
    print(train_data_path)
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path,
        correlation_rate=correlation_rate,
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path,
        correlation_rate=correlation_rate,
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=mnist_location+'/images/',
        resampling=False,
        root_dir=mnist_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path,
        correlation_rate=correlation_rate,
    )
    
    return train_dl, valid_dl, test_dl


def generate_data_loaders_chexpert(suffix):
    """Generate the train and validation dataloaders for the MNIST dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    
    if suffix in ["_model_robustness","_model_responsiveness"]:
        suffix = ""
    
    chexpert_location = '../../../datasets/chexpert'
    train_data_path = chexpert_location+'/preprocessed/train.pkl'
    valid_data_path = chexpert_location+'/preprocessed/val.pkl'
    test_data_path = chexpert_location+'/preprocessed/test.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=chexpert_location+'/images/',
        resampling=False,
        root_dir=chexpert_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=chexpert_location+'/images/',
        resampling=False,
        root_dir=chexpert_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=chexpert_location+'/images/',
        resampling=False,
        root_dir=chexpert_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    return train_dl, valid_dl, test_dl

def generate_data_loaders_dsprites(suffix):
    """Generate the train and validation dataloaders for the MNIST dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """
    
    if suffix in ["_model_robustness","_model_responsiveness"]:
        suffix = ""
    
    dsprites_location = '../../../datasets/dsprites'
    train_data_path = dsprites_location+'/preprocessed/train.pkl'
    valid_data_path = dsprites_location+'/preprocessed/val.pkl'
    test_data_path = dsprites_location+'/preprocessed/test.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=dsprites_location+'/images/',
        resampling=False,
        root_dir=dsprites_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=dsprites_location+'/images/',
        resampling=False,
        root_dir=dsprites_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=dsprites_location+'/images/',
        resampling=False,
        root_dir=dsprites_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    return train_dl, valid_dl, test_dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CEM Concept Vectors')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment we plan to run. Valid names include mnist, cub, and xor' )
    parser.add_argument('--num_gpus',type=int,
                        help='Number of GPUs to use when training',
                        default=0)
    parser.add_argument('--num_epochs',type=int,default=1,help='How many epochs to train for')
    parser.add_argument('--validation_epochs',type=int,default=1,help='How often should we run the validation script')
    parser.add_argument('--seed',type=int,default=42,help='Random seed for training')
    parser.add_argument('--num_workers',type=int,default=8,help='Number of workers')
    parser.add_argument('--sample_train',type=float,default=1.0,help='Fraction of the train dataset to sample')
    parser.add_argument('--sample_valid',type=float,default=1.0,help='Fraction of the valid dataset to sample')    
    parser.add_argument('--sample_test',type=float,default=1.0,help='Fraction of the test dataset to sample')    
    parser.add_argument('--concept_pair_loss_weight',type=float,default=0,help='Weight for the concept pair loss in the loss')
    parser.add_argument('--lr',type=float,default=0.01,help='Learning Rate for training')
    parser.add_argument('--weight_decay',type=float,default=4e-05,help="Weight decay regularization")
    parser.add_argument('--concept_loss_weight',type=float,default=5.0,help="How much emphasis to place on concept accuracy")
    parser.add_argument('--correlation_rate',type=float,default=-1.0,help="How much to correlate number, color")

    args = parser.parse_args()
    data_by_correlation = {}
    
    experiment_name = args.experiment_name
    num_gpus = args.num_gpus
    num_epochs = args.num_epochs
    validation_epochs = args.validation_epochs
    seed = args.seed
    num_workers = args.num_workers
    lr = args.lr 
    weight_decay = args.weight_decay
    concept_loss_weight = args.concept_loss_weight
    correlation_rate = args.correlation_rate

    for seed in [42,43,44]:
        for correlation_rate in [0,0.2,0.4,0.6,0.8,1.0]:
            print("On seed {} correlation rate {}".format(seed,correlation_rate))
            if correlation_rate not in data_by_correlation:
                data_by_correlation[correlation_rate] = []

            pl.seed_everything(seed, workers=True)
            random.seed(seed)
            
            trainer = pl.Trainer(
                    gpus=num_gpus,
            )
            
            experiment_name_split = experiment_name.split("_")
            experiment_name = experiment_name_split[0]
            if len(experiment_name_split)>1:
                suffix = "_"+"_".join(experiment_name_split[1:])
            else:
                suffix = ""
            
            existing_weights = ''
            if suffix == '_model_robustness':
                existing_weights = 'resnet_model_robustness.pt'
            elif suffix == '_model_responsiveness':
                existing_weights = 'resnet_model_responsiveness.pt'
            
            if experiment_name == "xor":
                train_dl, valid_dl = generate_data_loaders_xor()
                n_concepts = 2
                n_tasks = 2
            elif experiment_name == "cub":
                train_dl, valid_dl, test_dl = generate_data_loaders_cub(suffix)
                n_concepts = 112
                n_tasks = 200
            elif experiment_name == "mnist":
                train_dl, valid_dl, test_dl = generate_data_loaders_mnist(suffix)
                n_concepts = 20
                n_tasks = 10
            elif experiment_name == "mnistcorrelated":
                train_dl, valid_dl, test_dl = generate_data_loaders_mnist_correlated(suffix,correlation_rate)
                n_concepts = 20
                n_tasks = 10
            elif experiment_name == "chexpert":
                train_dl, valid_dl, test_dl = generate_data_loaders_chexpert(suffix)
                n_concepts = 13
                n_tasks = 2
            elif experiment_name == "dsprites":
                train_dl, valid_dl, test_dl = generate_data_loaders_dsprites(suffix)
                n_concepts = 18
                n_tasks = 100
            else:
                print("{} is not a valid experiment name".format(experiment_name))


            if args.sample_train < 1.0:
                train_dataset = train_dl.dataset
                train_size = round(len(train_dataset)*args.sample_train)
                train_subset = random.sample(range(0,len(train_dataset)),train_size)
                train_dataset = torch.utils.data.Subset(train_dataset, train_subset)
                train_dl = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True, 
                                                    drop_last=True, num_workers=args.num_workers)
                
            if args.sample_valid < 1.0:
                valid_dataset = valid_dl.dataset
                valid_size = round(len(valid_dataset)*args.sample_valid)
                valid_subset = random.sample(range(0,len(valid_dataset)),valid_size)
                valid_dataset = torch.utils.data.Subset(valid_dataset, valid_subset)
                valid_dl = torch.utils.data.DataLoader(valid_dataset,batch_size=64, shuffle=False, 
                                                    drop_last=False, num_workers=args.num_workers)

            if args.sample_test < 1.0:
                test_dataset = test_dl.dataset
                test_size = round(len(test_dataset)*args.sample_test)
                test_subset = random.sample(range(0,len(test_dataset)),test_size)
                test_dataset = torch.utils.data.Subset(test_dataset, test_subset)
                test_dl = torch.utils.data.DataLoader(test_dataset,batch_size=64, shuffle=False, 
                                                    drop_last=False, num_workers=args.num_workers)


            if experiment_name == "xor":
                extractor_arch = c_extractor_arch
                imbalance = None
            elif experiment_name == "cub":
                extractor_arch = "resnet34"
                cub_location = '../../../datasets/CUB{}'.format(suffix)
                train_data_path = cub_location+'/preprocessed/train.pkl'
                imbalance = find_class_imbalance(train_data_path, True)
            elif experiment_name == 'mnist' or experiment_name=='mnistcorrelated':
                extractor_arch = "resnet34"
                train_data_path = "../../../datasets/colored_mnist/preprocessed/train.pkl"
                imbalance = find_class_imbalance(train_data_path, True)
            elif experiment_name == 'chexpert':
                extractor_arch = "resnet34"
                train_data_path = "../../../datasets/chexpert/preprocessed/train.pkl"
                imbalance = find_class_imbalance(train_data_path, True)
            elif experiment_name == 'dsprites':
                extractor_arch = "resnet34"
                train_data_path = "../../../datasets/dsprites/preprocessed/train.pkl"
                imbalance = None


            config = dict(
                cv=5,
                max_epochs=num_epochs,
                patience=15,
                batch_size=128,
                num_workers=num_workers,
                emb_size=16,
                extra_dims=0,
                concept_loss_weight=concept_loss_weight,
                normalize_loss=False,
                learning_rate=lr,
                weight_decay=weight_decay,
                weight_loss=True,
                pretrain_model=True,
                c_extractor_arch=extractor_arch,
                optimizer="sgd",
                bool=False,
                early_stopping_monitor="val_loss",
                early_stopping_mode="min",
                early_stopping_delta=0.0,
                sampling_percent=1,
                momentum=0.9,
                validation_epochs=validation_epochs,
                shared_prob_gen=False,
                sigmoidal_prob=False,
                sigmoidal_embedding=False,
                training_intervention_prob=0.25,
                embeding_activation=None,
                concat_prob=False,
                seed=seed,
                concept_pair_loss_weight = args.concept_pair_loss_weight,
                existing_weights=""
            )
            config["architecture"] = "ConceptEmbeddingModel"
            config["extra_name"] = f"New"
            config["shared_prob_gen"] = True
            config["sigmoidal_prob"] = True
            config["sigmoidal_embedding"] = False
            config['training_intervention_prob'] = 0.25 
            config['concat_prob'] = False
            config['emb_size'] = config['emb_size']
            config["embeding_activation"] = "leakyrelu"  
            config["check_val_every_n_epoch"] = validation_epochs

            test_results = training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=valid_dl,
                test_dl=test_dl,
                split=0,
                result_dir="results/{}".format(experiment_name),
                rerun=False,
                project_name='concept_hierarchies_{}'.format(experiment_name),
                seed=seed,
                activation_freq=0,
                single_frequency_epochs=0,
                imbalance=imbalance,
            )[1]

            results_by_intervened_idx = {}
            results_by_intervened_idx[0] = {'test_y_accuracy': test_results['test_y_accuracy'], 'test_c_accuracy': test_results['test_c_accuracy']}
            print("Results {}".format(results_by_intervened_idx))

            for intervened_idxs in [4,8,12,16,20]:
                config["intervention_idxs"] = sorted(random.sample(list(range(n_concepts)),intervened_idxs))
                test_results = training.train_model(
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    config=config,
                    train_dl=train_dl,
                    val_dl=valid_dl,
                    test_dl=test_dl,
                    split=0,
                    result_dir="results/{}".format(experiment_name),
                    rerun=False,
                    project_name='concept_hierarchies_{}'.format(experiment_name),
                    seed=seed,
                    activation_freq=0,
                    single_frequency_epochs=0,
                    imbalance=imbalance,
                )[1]
                results_by_intervened_idx[intervened_idxs] = {'test_y_accuracy': test_results['test_y_accuracy'], 'test_c_accuracy': test_results['test_c_accuracy']}
                print("Results {}".format(results_by_intervened_idx))
            data_by_correlation[correlation_rate].append(results_by_intervened_idx)
            os.system("rm results/mnistcorrelated/ConceptEmbeddingModelNew_resnet34_fold_1.pt")
    json.dump(data_by_correlation,open("../results/evaluation/mnist_correlation/correlation_data.json","w"))

#     cem_model = cem_train.construct_model(
#         n_concepts=n_concepts,
#         n_tasks=n_tasks,
#         config=og_config,
#         imbalance=None,
#         intervention_idxs=None,
#         adversarial_intervention=None,
#         active_intervention_values=None,
#         inactive_intervention_values=None,
#         c2y_model=False,
#     )
        
# #     cem_model = ConceptEmbeddingModel(
# #       n_concepts=n_concepts, # Number of training-time concepts
# #       n_tasks=n_tasks, # Number of output labels
# #       emb_size=16,
# #       concept_loss_weight=0.1,
# #       concept_pair_loss_weight = args.concept_pair_loss_weight,
# #       learning_rate=0.01,
# #       optimizer="adam",
# #       c_extractor_arch=extractor_arch, # Replace this appropriately
# #       training_intervention_prob=0.25, # RandInt probability
# #       experiment_name=experiment_name+suffix, 
# #       seed=seed, 
# #       existing_weights = existing_weights,
# #     )
    
#     trainer = pl.Trainer(
#         gpus=num_gpus,
#         max_epochs=num_epochs,
#         check_val_every_n_epoch=validation_epochs,
#     )

#     trainer.fit(cem_model, train_dl, valid_dl)
#     cem_model.write_concepts()
    
#     torch.save(cem_model.state_dict(), "cem_model.pt")

    
