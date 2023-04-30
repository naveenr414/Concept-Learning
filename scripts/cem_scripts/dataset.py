import numpy as np
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import DataLoader, Dataset
from copy import deepcopy
import random
import pickle
from torch.utils.data import DataLoader, Subset, random_split
import glob
from PIL import Image
import torch_geometric
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import torch
from experiments import hierarchy
from experiments.cbm_models import *

def to_tensor_dataset(x,y,c):
    tensor_x = torch.Tensor(np.moveaxis(x, -1,1))
    tensor_y = torch.Tensor(y).long()
    tensor_c = torch.Tensor(c)

    tensor_x /= 255
    dataset = TensorDataset(tensor_x,tensor_y,tensor_c)
    
    return dataset

def convert_c_matrix(c_vals,num_groups):
    if len(c_vals) != 16*112:
        # Embeddings for c_vals 
        all_embeddings = []
        for i,val in enumerate(c_vals):
            initial = []
            initial += [0 if j != i else val for j in range(len(c_vals))]
            all_embeddings.append(initial)
                            
        return torch.Tensor(all_embeddings)
    else:
        return c_vals.reshape((112,16))

def tensor_to_graph(dataset, edge_attr, edge_index):
    x = torch.stack([sample[0] for sample in dataset])
    y = torch.stack([sample[1] for sample in dataset])
    c = torch.stack([sample[2] for sample in dataset])
    
    num_nodes = torch.max(edge_index)+1
    num_groups = num_nodes-112
        
    dataset = [Data(x=convert_c_matrix(c[i],num_groups),y=y[i],edge_index=edge_index,edge_attr=edge_attr,batch=torch.Tensor([0]).long()) for i in range(len(c))]
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader

def tensor_to_graph_pretrain(dataset, edge_attr, edge_index):
    all_data = []
    for i in range(5):
        all_data += [sample[2] for sample in dataset]
    
    c = torch.stack(all_data)
        
    num_nodes = torch.max(edge_index)+1
    num_attributes = 112
    num_groups = num_nodes-num_attributes
        
    predicted_nodes = [random.randint(0,num_attributes-1) for i in range(len(c))]
    y = []
    c_values = [convert_c_matrix(c[i],num_groups) for i in range(len(c))]
    
    for i in range(len(c)):
        # Mask the appropriate value
#         y.append(float(c_values[i][predicted_nodes[i]][-1].detach()))
#         c_values[i][predicted_nodes[i]][-1] = 0
        y.append(float(c_values[i][predicted_nodes[i]][predicted_nodes[i]].detach()))
        c_values[i][predicted_nodes[i]][predicted_nodes[i]] = 0 

    y = torch.Tensor(y)
        
    dataset = [Data(x=c_values[i],y=y[i],masked_val=predicted_nodes[i],edge_index=edge_index,edge_attr=edge_attr,batch=torch.Tensor([0]).long()) for i in range(len(c_values))]
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader



def tensor_to_graph_var_var(dataset,edge_attr,edge_index):
    def convert_c_var_var(c_vals):
        return c_vals.reshape((len(c_vals),1))
    
    x = torch.stack([sample[0] for sample in dataset])
    y = torch.stack([sample[1] for sample in dataset])
    c = torch.stack([sample[2] for sample in dataset])
    
    dataset = [Data(x=convert_c_var_var(c[i]),y=y[i],edge_index=edge_index,edge_attr=edge_attr,batch=torch.Tensor([0]).long()) for i in range(len(c))]
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader

def tensor_to_graph_var_clause(dataset,edge_attr,edge_index,clauses):
    def convert_c_var_clause(c_vals):
        eye = torch.eye(len(c_vals))
        diag = torch.diag(c_vals)
        result = eye * diag    
        P = torch.zeros((len(result)+len(clauses), len(result)+len(clauses)))
        P[:len(result), :len(result)] = result
        
        return P
        
    x = torch.stack([sample[0] for sample in dataset])
    y = torch.stack([sample[1] for sample in dataset])
    c = torch.stack([sample[2] for sample in dataset])
    
    dataset = [Data(x=convert_c_var_clause(c[i]),y=y[i],edge_index=edge_index,edge_attr=edge_attr,batch=torch.Tensor([0]).long()) for i in range(len(c))]
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader


def load_mnist_independent():
    train = pickle.load(open("colored_mnist/images/train.pkl","rb"))
    val = pickle.load(open("colored_mnist/images/val.pkl","rb"))

    train_size = 10000
    val_size = 1000
    
    x_train = [np.array(Image.open(train[i]['img_path'])) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = [train[i]['attribute_label'] for i in range(train_size)] 
    
    x_val = [np.array(Image.open(val[i]['img_path'])) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = [val[i]['attribute_label'] for i in range(val_size)] 
    
    raise Exception("Need to develop predicted c values for MNIST")
    
def load_cub_independent():
    train = pickle.load(open("CUB/preprocessed/train.pkl","rb"))
    val = pickle.load(open("CUB/preprocessed/val.pkl","rb"))

    train_size = len(train)
    val_size = len(val)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = [train[i]['attribute_label'] for i in range(train_size)] 

    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/valid_c.npy","rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val)

def load_cub_independent_robust():
    train = pickle.load(open("CUB/preprocessed/train.pkl","rb"))
    val = pickle.load(open("CUB/preprocessed/val.pkl","rb"))

    train_size = len(train)
    val_size = len(val)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = [train[i]['attribute_label'] for i in range(train_size)] 

    for i in range(len(c_train)):
        for j in range(len(c_train[i])):
            if random.random() < 0.05:
                c_train[i][j] = 1-c_train[i][j]
    
    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/valid_c.npy","rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val)

def load_cub_fixed():
    train = pickle.load(open("CUB/preprocessed/train.pkl","rb"))
    val = pickle.load(open("CUB/preprocessed/val.pkl","rb"))
    test = pickle.load(open("CUB/preprocessed/test.pkl","rb"))

    train_size = len(train)
    val_size = len(val)
    test_size = len(test)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/train_c_fixed.npy","rb"))

    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/valid_c_fixed.npy","rb"))
    
    x_test = [np.zeros((224,224,3)) for i in range(test_size)]
    y_test = [test[i]['class_label'] for i in range(test_size)]
    c_test = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/test_c_fixed.npy","rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val), to_tensor_dataset(x_test, y_test, c_test)

def load_dataset(folder_name,fixed):
    suffix = ""
    
    if fixed:
        suffix = "_fixed"
    
    train = pickle.load(open("{}/preprocessed/train.pkl".format(folder_name),"rb"))
    val = pickle.load(open("{}/preprocessed/val.pkl".format(folder_name),"rb"))
    test = pickle.load(open("{}/preprocessed/test.pkl".format(folder_name),"rb"))

    train_size = len(train)
    val_size = len(val)
    test_size = len(test)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/{}/train_c{}.npy".
                           format(folder_name,suffix),"rb"))

    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/{}/valid_c{}.npy".
                         format(folder_name,suffix),"rb"))
    
    x_test = [np.zeros((224,224,3)) for i in range(test_size)]
    y_test = [test[i]['class_label'] for i in range(test_size)]
    c_test = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/{}/test_c{}.npy".
                          format(folder_name,suffix),"rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val), to_tensor_dataset(x_test, y_test, c_test)


def load_cub_sequential():
    train = pickle.load(open("CUB/preprocessed/train.pkl","rb"))
    val = pickle.load(open("CUB/preprocessed/val.pkl","rb"))
    test = pickle.load(open("CUB/preprocessed/test.pkl","rb"))

    train_size = len(train)
    val_size = len(val)
    test_size = len(test)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/train_c.npy","rb"))

    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/valid_c.npy","rb"))
    
    x_test = [np.zeros((224,224,3)) for i in range(test_size)]
    y_test = [test[i]['class_label'] for i in range(test_size)]
    c_test = np.load(open("/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck/results/CUB/test_c.npy","rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val), to_tensor_dataset(x_test, y_test, c_test)


def load_cub_cem():
    train = pickle.load(open("CUB/preprocessed/train.pkl","rb"))
    val = pickle.load(open("CUB/preprocessed/val.pkl","rb"))

    train_size = len(train)
    val_size = len(val)

    x_train = [np.zeros((224,224,3)) for i in range(train_size)]
    y_train = [train[i]['class_label'] for i in range(train_size)]
    c_train = np.load(open("/home/njr61/rds/hpc-work/cem/cem/results/cem_train.npy","rb"))

    x_val = [np.zeros((224,224,3)) for i in range(val_size)]
    y_val = [val[i]['class_label'] for i in range(val_size)]
    c_val = np.load(open("/home/njr61/rds/hpc-work/cem/cem/results/cem_val.npy","rb"))
    
    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val)

def find_closest_vectors(matrix,num_indices=10):
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
        closest_indices.append([(j, sim) for j, d, sim in distances if j != i][:num_indices])

    return closest_indices

def get_related_scores(embedding_matrix,power=1,num_indices=10):
    closest_vectors = find_closest_vectors(embedding_matrix,num_indices=num_indices)
    related_concept2vec = {}
    identity_function = lambda s: s
    opposite_function = lambda s: 1-s

    for i in range(112):
        related_concept2vec[i] = {}

        for index,similarity in closest_vectors[i]:
            confidence = abs(similarity)**power
            if similarity > 0:
                related_concept2vec[i][index] = (identity_function,confidence)
            else:
                related_concept2vec[i][index] = (opposite_function,confidence)
                
    return related_concept2vec

def get_attr_index(groups,attributes,sim_matrix,use_random=False,fully_connected=False):
    indexes = [[attributes.index(i) for i in group] for group in groups]
    edge_index = []

    index_to_group_num = {}

    if use_random:
        random_groups = list(range(len(attributes)))
        random.shuffle(random_groups)
        random_indexes = []

        start = 0
        for group_num, group in enumerate(indexes):
            group_length = len(group)            
            random_indexes.append(random_groups[start:start+group_length])
            start += group_length
        indexes = random_indexes

    if fully_connected:
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix)):
                edge_index.append((i,j))
    else:        
        for group_num,group in enumerate(indexes): 
            for i in group:
                index_to_group_num[i] = group_num
                for j in group:
                    edge_index.append((i,j))
                    
                
    edge_attr = []
    for (i,j) in edge_index:
        edge_weight = sim_matrix[i,j]
        edge_attr.append([edge_weight])

    edge_attr = torch.Tensor(edge_attr)    
    edge_index = torch.Tensor(edge_index).long().T
    
    return edge_attr, edge_index

def create_var_var_graph(clauses):
    edge_index = []
    edge_attr = []
    
    for clause in clauses:
        for var1 in clause:
            for var2 in clause:
                edge_index.append((var1[0],var2[0]))
                
    for i,clause in enumerate(clauses):
        embedding = [0 for i in range(len(clauses))]
        embedding[i] = 1
        
        for var1 in clause:
            for var2 in clause:
                edge_attr.append(embedding)
                
    edge_attr = torch.Tensor(edge_attr)    
    edge_index = torch.Tensor(edge_index).long().T
    
    return edge_attr, edge_index

def create_var_clause_graph(clauses):
    edge_index = []
    edge_attr = []
    num_variables = len(clauses)*len(clauses[0])
    
    for i,clause in enumerate(clauses):
        for var1 in clause:
            clause_idx = i+num_variables
            edge_index.append((var1[0],clause_idx))
                
    edge_attr = []
    for (i,j) in edge_index:
        edge_weight = 1
        edge_attr.append([edge_weight])

    edge_attr = torch.Tensor(edge_attr)    
    edge_index = torch.Tensor(edge_index).long().T
    
    return edge_attr, edge_index

def create_fully_connected_graph(n):
    edge_index = []
    edge_attr = []
    
    for i in range(n):
        for j in range(n):
            edge_index.append((i,j))
                
    edge_attr = []
    for (i,j) in edge_index:
        edge_weight = 1
        edge_attr.append([edge_weight])

    edge_attr = torch.Tensor(edge_attr)    
    edge_index = torch.Tensor(edge_index).long().T
    
    return edge_attr, edge_index


def get_attr_index_cem(groups,attributes,sim_matrix,use_random=False):
    indexes = [[attributes.index(i) for i in group] for group in groups]
    edge_index = []

    index_to_group_num = {}

    if use_random:
        random_groups = list(range(len(attributes)))
        random.shuffle(random_groups)
        random_indexes = []

        start = 0
        for group_num, group in enumerate(indexes):
            group_length = len(group)
            random_indexes.append(random_groups[start:start+group_length])
            start += group_length
        indexes = random_indexes

    
    for group_num,group in enumerate(indexes): 
        for i in group:
            index_to_group_num[i] = group_num
            for j in group:
                edge_index.append((i,j))

    edge_attr = []
    for (i,j) in edge_index:
        edge_weight = sim_matrix[i,j]
        edge_attr.append([edge_weight])

    edge_attr = torch.Tensor(edge_attr)    
    edge_index = torch.Tensor(edge_index).long().T
    
    return edge_attr, edge_index


def evaluate_3_sat(clauses,dataset):
    num_datapoints = len(dataset)
    num_clauses = len(clauses)
    results = []
    for i in range(num_clauses):
        clause_value = 0
        for var, coeff in clauses[i]:
            var_value = dataset[var]
            if coeff == -1:
                var_value = 1-var_value
            clause_value |= var_value
        if clause_value == 0:
            return 0
    return 1


def create_random_3_sat(num_variables):
    variable_groups = []
    var_indices = list(range(num_variables))
    while len(var_indices) >= 3:
        group = random.sample(var_indices, 3)
        variable_groups.append(group)
        var_indices = [var for var in var_indices if var not in group]
        
    clauses = []

    for group in variable_groups:
        group_coefficients = [random.choice([-1, 1]) for _ in group]
        clauses.append([(var, coeff) for var, coeff in zip(group, group_coefficients)])

    return clauses

def create_3_sat_dataset(num_variables, train_data, val_data):
    random_3_sat = create_random_3_sat(num_variables)
    
    x_train = []
    c_train = []
    y_train = []
    
    x_val = []
    c_val = []
    y_val = []
    
    for i in range(train_data):
        x_train.append(np.zeros(1))
        c_train.append(np.random.choice([0, 1], size=num_variables))
        y_train.append(evaluate_3_sat(random_3_sat,c_train[-1]))
        
    for i in range(val_data):
        x_val.append(np.zeros(1))
        c_val.append(np.random.choice([0, 1], size=num_variables))
        y_val.append(evaluate_3_sat(random_3_sat,c_val[-1]))

    return to_tensor_dataset(x_train, y_train, c_train), to_tensor_dataset(x_val, y_val, c_val), random_3_sat

def create_sim_matrix(groups):
    num_variables = len(groups)*len(groups[0])
    sim_matrix = np.zeros((num_variables,num_variables))
    
    for group in groups:
        for i in group:
            for j in group:
                sim_matrix[i,j] = 1
                
    return sim_matrix

def create_hierarchy(groups):
    h = hierarchy.Hierarchy()
    
    current_num = 0
    num_groups = len(groups)

    embedding_list = [[0 for i in range(num_groups)] for i in range(num_groups*len(groups[0]))]
        
    for group in groups:
        for num in group:
            embedding_list[num][current_num] = 1 
        current_num += 1
        
    embedding_list = np.array(embedding_list)
    
def update_hyperparameters_graph(model_type,hyperparameters):    
    if 'clauses' in hyperparameters:
        clauses = hyperparameters['clauses']
        sat_group_idx = [[j[0] for j in i] for i in clauses]
        sat_group = [[str(j) for j in i] for i in sat_group_idx]
        sat_attributes = [str(i) for i in range(len(clauses)*len(clauses[0]))]
        sat_sim_matrix = create_sim_matrix(sat_group_idx)
    
    if 'sim_matrix' not in hyperparameters:
        hyperparameters['sim_matrix'] = sat_sim_matrix
        
    if 'indexes' not in hyperparameters:
        hyperparameters['indexes'] = sat_group_idx

    if 'attributes' not in hyperparameters: 
        hyperparameters['attributes'] = sat_attributes
        
    if 'group' not in hyperparameters:
        hyperparameters['groups'] = sat_group
    
    if 'mlp' in model_type:
        hyperparameters['in_dim'] = sum([len(i) for i in hyperparameters['indexes']])
        hyperparameters['bottleneck_size'] = hyperparameters['in_dim']
        if hyperparameters['bottleneck_size'] == 112:
            hyperparameters['out_dim'] = 200
        elif hyperparameters['bottleneck_size'] == 18:
            hyperparameters['out_dim'] = 100
        else:
            hyperparameters['out_dim'] = 2
        
    if model_type == 'gnn_bool':
        hyperparameters['edge_dim'] = 1
        hyperparameters['in_dim'] = 1
    
    return hyperparameters

def get_dataset_graph_cub_pretrain(model_type,groups,attributes,sim_matrix,train,val,test,use_random=False):
    edge_attr, edge_index = get_attr_index(groups,attributes,sim_matrix,use_random=use_random) 
    train = tensor_to_graph_pretrain(train,edge_attr,edge_index)
    val = tensor_to_graph_pretrain(val,edge_attr,edge_index)
    test = tensor_to_graph_pretrain(test,edge_attr,edge_index)
    return edge_attr, edge_index, train, val, test


def get_dataset_graph_cub(model_type,groups,attributes,sim_matrix, train, val,test,use_random=False): 
    edge_attr, edge_index = get_attr_index(groups,attributes,sim_matrix,use_random=use_random) 
    if model_type == 'gnn_bool':
        train = tensor_to_graph_var_var(train,edge_attr,edge_index)
        val = tensor_to_graph_var_var(val,edge_attr,edge_index)
        test = tensor_to_graph_var_var(test,edge_attr,edge_index)
    else:
        train = tensor_to_graph(train,edge_attr,edge_index)
        val = tensor_to_graph(val,edge_attr,edge_index)
        test = tensor_to_graph(test,edge_attr,edge_index)
    return edge_attr, edge_index, train, val, test
    
def get_dataset_graph(model_type,clauses,sat_train,sat_val):
    if model_type == 'gnn_bool':
        sat_edge_attr, sat_edge_index = create_fully_connected_graph(len(clauses))
        sat_train_graph = tensor_to_graph_var_var(sat_train,sat_edge_attr,sat_edge_index)
        sat_val_graph = tensor_to_graph_var_var(sat_val,sat_edge_attr,sat_edge_index)
    
    elif model_type == 'gnn_basic' or model_type == 'gnn' or model_type == 'gnn_gat':
        sat_group_idx = [[j[0] for j in i] for i in clauses]
        sat_group = [[str(j) for j in i] for i in sat_group_idx]
        sat_attributes = [str(i) for i in range(len(clauses)*len(clauses[0]))]
        sat_sim_matrix = create_sim_matrix(sat_group_idx)
        
        sat_edge_attr, sat_edge_index = get_attr_index(sat_group,sat_attributes,sat_sim_matrix)
    
        sat_train_graph = tensor_to_graph(sat_train,sat_edge_attr,sat_edge_index)
        sat_val_graph = tensor_to_graph(sat_val,sat_edge_attr,sat_edge_index)
    
    elif model_type == 'gnn_var':
        sat_edge_attr, sat_edge_index = create_var_var_graph(clauses)
        
        sat_train_graph = tensor_to_graph_var_var(sat_train,sat_edge_attr,sat_edge_index,clauses)
        sat_val_graph = tensor_to_graph_var_var(sat_val,sat_edge_attr,sat_edge_index,clauses)
    
    elif model_type == 'gnn_clause':
        sat_edge_attr, sat_edge_index = create_var_clause_graph(clauses)
        
        sat_train_graph = tensor_to_graph_var_clause(sat_train,sat_edge_attr,sat_edge_index,clauses)
        sat_val_graph = tensor_to_graph_var_clause(sat_val,sat_edge_attr,sat_edge_index,clauses)
    return sat_edge_attr, sat_edge_index, sat_train_graph, sat_val_graph
