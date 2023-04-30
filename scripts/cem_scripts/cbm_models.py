import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, global_add_pool
from torch_scatter import scatter
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from torch_geometric.nn import GATConv
from pytorch_lightning.callbacks import Callback
import secrets
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score



def diff_xor(x,y):
    return torch.mul(torch.sign(torch.abs(x - y)), 0.5) + 0.5


class Decoder_CBM(pl.LightningModule):
    def __init__(self,model_type,hyperparameters,use_wandb):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.lr = hyperparameters['lr']
        self.groups = hyperparameters['indexes']
        self.attributes = hyperparameters['attributes']
        self.emb_size = hyperparameters['emb_dim']
        self.bottleneck_size = hyperparameters['bottleneck_size']
        self.output_classes = hyperparameters['out_dim']
        self.model_type = model_type 
        self.use_wandb = use_wandb
        self.hyperparameters = hyperparameters
        
        if model_type == 'mlp_group':
            self.group_weights = {}
            for group in self.groups:
                self.group_weights[' '.join([str(j) for j in group])] = torch.nn.Sequential(torch.nn.Linear(len(group),self.emb_size),
                                                                                            torch.nn.ReLU(),
                                                                                            torch.nn.Linear(self.emb_size,self.emb_size))
            self.sorted_keys = sorted(self.group_weights.keys())

        if model_type == 'mlp':
            self.fc = torch.nn.Sequential(torch.nn.Linear(self.bottleneck_size,self.emb_size),
                                          torch.nn.ReLU(),torch.nn.Linear(self.emb_size,self.output_classes))
        elif model_type == 'mlp_group':
            self.fc = torch.nn.Sequential(torch.nn.ReLU(),torch.nn.Linear(self.emb_size*len(self.groups),self.emb_size),
                                          torch.nn.ReLU(),torch.nn.Linear(self.emb_size,self.output_classes))
        elif model_type == 'mlp_3sat':
            self.get_weights = torch.nn.Sequential(torch.nn.Linear(self.bottleneck_size,self.emb_size),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.emb_size,self.bottleneck_size))
        elif model_type == 'mlp_bool':
            self.branches = []
            self.fc = torch.nn.Linear(len(self.groups),self.output_classes)
            
            for i in range(len(self.groups)):
                self.branches.append(nn.Sequential(
                    nn.Linear(len(self.groups[i]), 5),
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, 1),
                    nn.Sigmoid()
                ))
        else:
            raise Exception("No model found named {}".format(self.model_Type))

    
    def forward(self, x):
        if self.model_type == 'mlp':
            y = self.fc(x)
        elif self.model_type == 'mlp_group':
            self.intermediate_values = {}
        
            for group_name in self.group_weights:
                used_attributes = group_name.split(" ")
                indices = [int(attribute) for attribute in used_attributes]
                concat_rep = torch.stack([x[:,index] for index in indices]).T

                output_value = self.group_weights[group_name](concat_rep)
                self.intermediate_values[group_name] = output_value


            intermediate_representation = torch.stack([self.intermediate_values[i] for i in self.sorted_keys])
            batch_size = intermediate_representation.shape[1]
            num_groups = intermediate_representation.shape[0]
            emb_size = intermediate_representation.shape[2]
            
            intermediate_representation = intermediate_representation.view(batch_size,num_groups*emb_size)
            y = self.fc(intermediate_representation)
        elif self.model_type == 'mlp_3sat':
            weights = self.get_weights(x)
            scaled_values = torch.stack([diff_xor(weights[i],x[i]) for i in range(len(x))])

            clause_values = []
            for c in self.groups:
                clause_values.append(torch.stack([scaled_values[:,j] for j in c]))
            clause_values = torch.stack(clause_values)
            reduced = torch.zeros((clause_values.shape[0],clause_values.shape[2]))
            for i in range(clause_values.shape[0]):
                for j in range(clause_values.shape[2]):
                    reduced[i][j] = torch.max(clause_values[i,:,j])

            predictions = torch.ones((reduced.shape[1]))
            for i in range(reduced.shape[1]):
                predictions[i] = torch.min(reduced[:,i])
            predictions = torch.clip(predictions,0,1)
            y = torch.stack([1-predictions,predictions]).T
        elif self.model_type == 'mlp_bool':
            branch_outputs = []
            for i in range(len(self.groups)):
                branch_outputs.append(self.branches[i](x[:, self.groups[i]]))
            concatenated_output = torch.cat((branch_outputs), dim=1)
            y = self.fc(concatenated_output)
            
        return y
    
    def batch_end(self,batch,batch_idx,name):
        x, y, c = batch
        y = y.long()
        y_hat = self.forward(c)
        preds = torch.argmax(y_hat,dim=1)
        acc = sum(preds == y)/len(preds)
        loss_task = torch.nn.CrossEntropyLoss()(y_hat, y)
        
        if name == 'train' and self.use_wandb:
            self.log("loss",loss_task)
            wandb.log({"loss": loss_task})
        
        return {'loss': loss_task, 'acc': acc, "num_preds": len(preds)}
    
    def epoch_end(self,outputs,name):
        num_datapoints = sum([x["num_preds"] for x in outputs])
        avg_train_loss = sum([x["loss"].detach()*x["num_preds"] for x in outputs])
        avg_train_loss /= num_datapoints
        
        avg_acc = sum([float(x["acc"].detach()*x["num_preds"]) for x in outputs])
        avg_acc /= num_datapoints
        
        if self.use_wandb:
            self.log('{}_loss'.format(name), avg_train_loss)
            self.log('{}_acc'.format(name),avg_acc)

    def training_step(self, batch, batch_idx):
        return self.batch_end(batch,batch_idx,"train")
    
    def training_epoch_end(self, training_step_outputs):
        self.epoch_end(training_step_outputs,"train")

    def validation_step(self,batch,batch_idx):
        return self.batch_end(batch,batch_idx,"val")       
        
    def validation_epoch_end(self, training_step_outputs):
        self.epoch_end(training_step_outputs,"val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hyperparameters['epochs'])
        return [optimizer], [scheduler]
            
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=1, edge_dim=1, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Args:
            h_i: (num_edges, node_dimension) - destination node features, essentially h[edge_index[0]]
            h_j: (num_edges, node_dimension) - source node features, essentially h[edge_index[1]]
            edge_attr: (num_edges, edge_features) - edge features
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)

        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out) 

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    
class MPNNModel(Module):
    def __init__(self, model_type, hyperparameters,use_wandb,pretrain=False):
        super().__init__()
        self.in_dim = hyperparameters['in_dim']
        self.groups = hyperparameters['indexes']
        self.emb_dim = hyperparameters['emb_dim']
        self.edge_dim = hyperparameters['edge_dim']
        self.out_dim = hyperparameters['out_dim'] 
        self.num_layers = hyperparameters['num_layers']
        self.model_type = model_type
        self.num_groups = len(self.groups)
        self.use_wandb = use_wandb
        
        self.pretrain = pretrain
        self.pretrain_output = Linear(self.emb_dim,1)
        
        self.lin_in = Linear(self.in_dim, self.emb_dim)
        
        self.convs = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if self.model_type == 'gnn_gat':
                self.convs.append(GATConv(self.emb_dim, self.edge_dim))
            else:
                self.convs.append(MPNNLayer(self.emb_dim, self.edge_dim, aggr='add'))
                
        if self.model_type == 'gnn_bool':
            self.branches = nn.ModuleList()

            for i in self.groups:
                self.branches.append(nn.Sequential(
                    nn.Linear(len(i), 5),
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, self.in_dim),
                    nn.Sigmoid()
                ))
        elif self.model_type == 'gnn_bool_cem':
            self.branches = nn.ModuleList()

            for i in self.groups:
                self.branches.append(nn.Sequential(
                    nn.Linear(len(i)*16, 5),
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, self.in_dim),
                    nn.Sigmoid()
                ))

        
        self.pool = global_mean_pool
        
        self.group_pred = Linear(self.emb_dim,self.emb_dim)
        self.whole_pred = Linear(self.emb_dim*self.num_groups,self.emb_dim)
        self.lin_pred = Linear(self.emb_dim, self.out_dim)
        self.dropout = nn.Dropout(p=0.5)
        
    def get_embeddings(self,data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        if self.model_type == 'gnn_bool_cem':
            h, edge_attr, edge_index = self.get_h_bool(data,cem=True)
        elif data.x.shape[1] == 16: # CEM Models 
            h = data.x
        elif self.model_type == 'gnn_bool':
            h, edge_attr, edge_index = self.get_h_bool(data)            
        else:
            h = self.lin_in(data.x)
            
            
            
        for conv in self.convs:
            if self.model_type == 'gnn_gat':
                h = conv(h, data.edge_index) + h
            else:
                h = conv(h, edge_index, edge_attr) + h
        return h
    
    def get_h_bool(self,data,cem=False):
        branch_outputs = []
        clause_size = sum([len(i) for i in self.groups])
        total_size = len(data.x)
        
        if cem:
            clause_size *= 16
            total_size *= 16
            x = data.x.reshape((total_size//clause_size,clause_size//16,16))
        else:
            x = data.x.reshape((len(data.x)//clause_size,clause_size))
        
        for i in range(len(self.groups)):
            if cem:
                inp = x[:,self.groups[i],:]
                inp = inp.reshape((len(inp),len(self.groups[i])*inp.shape[2]))
                branch_outputs.append(self.branches[i](inp))
            else:
                branch_outputs.append(self.branches[i](x[:, self.groups[i]]))
            
        h = torch.stack((branch_outputs), dim=1)
        h = h.view(-1, h.size(-1))
                
        h = self.lin_in(h)
        
        edge_attr = []
        edge_index = []
        
        for i in range(0,len(h),len(self.groups)):
            for add_1 in range(len(self.groups)):
                for add_2 in range(len(self.groups)):
                    edge_index.append((i+add_1,i+add_2))

                    
        for (i,j) in edge_index:
            edge_attr.append([1])
            
        edge_attr = torch.Tensor(edge_attr)    
        edge_index = torch.Tensor(edge_index).long().T  

        return h, edge_attr, edge_index
        
    def forward(self, data):
        h = self.get_embeddings(data)
        batch = data.batch
        
        if self.pretrain and (self.model_type == 'gnn_basic' or self.model_type == 'gnn'):
            masked_ids = data.masked_val

            for i in range(len(masked_ids)):
                num_nodes = 112
                masked_ids[i] += (num_nodes)*i 
                
            h_vals = h[masked_ids,:]
                        
            out = self.pretrain_output(h_vals)
            return out[:,0]
        
        if self.model_type == 'gnn_bool' or self.model_type == 'gnn_bool_cem':
            batch = []
            for i in range(len(h)//len(self.groups)):
                for j in range(len(self.groups)):
                    batch.append(i)
            batch = torch.Tensor(batch).long()
        
        if self.model_type == 'gnn_basic' or self.model_type == 'gnn_bool' or self.model_type == 'gnn_bool_cem':                        
            h_graph = self.pool(h, batch)
            
            out = self.lin_pred(h_graph)
            return out
        
        batch_size = torch.max(batch)+1    
        h_by_batch = h.reshape(batch_size,len(h)//batch_size,h.shape[1])
        h_by_group = torch.zeros(batch_size,len(self.groups),h.shape[1])
                
        for i in range(len(self.groups)):
            h_by_group[:,i,:] = torch.mean(h_by_batch[:,self.groups[i],:],dim=1)

        if torch.cuda.is_available():
            h_by_group = h_by_group.cuda()
            
        h_by_whole = self.whole_pred(h_by_group.view(batch_size, -1))
        out = self.lin_pred(h_by_whole)
        return out
    
def train(model, train_loader, optimizer, device, use_wandb=False, pretrain=False):
    model.train()
    loss_all = 0
    

    for data in train_loader:
        data = data.to(device)
        
        if torch.cuda.is_available():
            data = data.cuda()
        
        optimizer.zero_grad()
        y_pred = model(data)
               
        if pretrain:
            loss = F.mse_loss(y_pred,data.y)
        else:
            loss = F.cross_entropy(y_pred, data.y) 
        
        loss.backward(retain_graph=True)
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
                
        if use_wandb:
            wandb.log({'loss': loss})

    return loss_all / len(train_loader.dataset)


def eval_gnn(model, loader,pretrain=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    error = 0
    total = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            y_hat = model(data)
            
            if pretrain:
                losses.append(F.mse_loss(y_hat,data.y))
                error += 1
                total += 1
            else:
                losses.append(F.cross_entropy(y_hat, data.y))
                y_pred = torch.argmax(y_hat,axis=1)
                error += sum(y_pred != data.y)
                total += len(data.y)
            
    acc = 1-error/total
    loss = torch.mean(torch.stack(losses))
            
    return loss, acc


def initialize_model(model_type,hyperparameters,dataset,use_wandb=False,pretrain=False,weights={}):
    if 'gnn' in model_type:
        model = MPNNModel(model_type, hyperparameters,use_wandb=use_wandb,pretrain=pretrain)
        
        if weights != {}:
            model.load_state_dict(weights,strict=False)
        
    elif 'mlp' in model_type:
        model = Decoder_CBM(model_type, hyperparameters,use_wandb=use_wandb)
        
    if use_wandb:
        config = {}
        config["epochs"] = hyperparameters['epochs']
        config['architecture'] = model_type
        config['learning_rate'] = hyperparameters['lr']
        config['seed'] = hyperparameters['seed']
        config['hierarchy_name'] = hyperparameters['hierarchy_name']
        config['pretrain'] = hyperparameters['pretrain']
        
        if pretrain:
            config["pretrain"] = True
        
        if "clauses" in hyperparameters:
            config['num_clauses'] = len(hyperparameters['clauses'])

        config['dataset'] = dataset
            
        project_name = "cbm-with-hierarchy"
        random_name = secrets.token_hex(4)
        
        wandb.init(
            project=project_name,
            name=random_name,
            config=config
        )

        model.train()

        
    return model
        
def train_model(model,model_type,train_dataset,val_dataset,hyperparameters,verbose=False,use_wandb=False,pretrain=False): 
    if 'gnn' not in model_type:
        batch_size = 32
        if use_wandb:
            wandb_logger = WandbLogger()
            wandb_logger.watch(model)
            trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=hyperparameters['epochs'], enable_progress_bar=verbose,logger=wandb_logger, log_every_n_steps=batch_size)
        else:
            trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=hyperparameters['epochs'], enable_progress_bar=verbose, log_every_n_steps=batch_size)
        trainer.fit(model, DataLoader(train_dataset,batch_size=batch_size),DataLoader(val_dataset,batch_size=32))
        
        return model
    else:
        if use_wandb:
            wandb.watch(model, log='all')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters['epochs'])
    
        for epoch in range(1, 1+hyperparameters['epochs']):
            model.train()
            train(model, train_dataset, optimizer, device,use_wandb=use_wandb,pretrain=pretrain)
            
            if use_wandb:
                train_loss, train_acc = eval_gnn(model, train_dataset,pretrain=pretrain)
                val_loss, val_acc = eval_gnn(model, val_dataset,pretrain=pretrain)

                wandb.log({'train_acc': train_acc, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})
            scheduler.step()
            
        return model        
    
def eval_model(model,model_type, val_loader,pretrain=False):
    if 'gnn' not in model_type:
        y = torch.stack([sample[1] for sample in val_loader])
        c = torch.stack([sample[2] for sample in val_loader])
        
        predictions = model(c)
        softmaxed_logits = F.softmax(predictions, dim=1)
        loss = F.cross_entropy(softmaxed_logits, y)
        
        if len(predictions.shape) == 2:
            predictions = torch.argmax(predictions,dim=1)
            acc = float(sum(predictions == y)/len(y))
        else:
            binary_predictions = (predictions >= 0.5).int()
            acc = torch.mean((binary_predictions == y.int()).float()).item()
        
        if softmaxed_logits.shape[1] == 2:
            auc = roc_auc_score(y.cpu().numpy(), softmaxed_logits.detach().cpu().numpy()[:, 1])
            return loss, acc, auc
            
        return loss,acc

    else:
        return eval_gnn(model,val_loader,pretrain=pretrain)
    
def find_optimal_lr(model_type,lr_values,baseline_hyperparameters,train_dataset,val_dataset):
    hyperparameters = deepcopy(baseline_hyperparameters)
    score_by_lr = {}
    
    for lr in lr_values:
        hyperparameters['lr'] = lr
        model = initialize_model(model_type,hyperparameters)
        model = train_model(model,model_type,train_dataset,val_dataset,hyperparameters)
        score_by_lr[lr] = eval_model(model,model_type,train_dataset)
        
    best_lr = min(lr_values, key=lambda k: (score_by_lr[k],k))
    return best_lr
