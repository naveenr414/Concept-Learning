import torch
import pytorch_lightning as pl
from cem.models.cbm import ConceptBottleneckModel
import numpy as np
import os
import sklearn.metrics
from torchvision.models import resnet50
import random

################################################################################
## HELPER LAYERS
################################################################################


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


################################################################################
## OUR MODEL
################################################################################

def compute_accuracy(
    c_pred,
    y_pred,
    c_true,
    y_true,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        return compute_bin_accuracy(
            c_pred,
            y_pred,
            c_true,
            y_true,
        )
    c_pred = c_pred.reshape(-1).cpu().detach() > 0.5
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    c_true = c_true.reshape(-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    c_accuracy = sklearn.metrics.accuracy_score(c_true, c_pred)
    try:
        c_auc = sklearn.metrics.roc_auc_score(
            c_true,
            c_pred,
            multi_class='ovo',
        )
    except:
        c_auc = 0.0
    try:
        c_f1 = sklearn.metrics.f1_score(
            c_true,
            c_pred,
            average='macro',
        )
    except:
        c_f1 = 0
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovo',
        )
    except:
        y_auc = 0.0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    except:
        y_f1 = 0.0
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        experiment_name="default",
        emb_size=16,
        concept_loss_weight=1,
        concept_pair_loss_weight=0.1,
        task_loss_weight=1,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=False,
        pretrain_model=True,
        c_extractor_arch=resnet50,
        optimizer="adam",
        shared_prob_gen=True, #False,
        top_k_accuracy=2,
        intervention_idxs=None,
        sigmoidal_prob=True, #False,
        sigmoidal_embedding=False,
        adversarial_intervention=False,
        training_intervention_prob=0.25, #0.0,
        active_intervention_values=None,
        inactive_intervention_values=None,
        embeding_activation="leakyrelu", # None
        concat_prob=False,
        existing_weights='',
        gpu=int(torch.cuda.is_available()),
        seed=-1,
        related_concepts=None,
        vae_model = None,
        stratified_model = False,
        limit=None,
    ):        
        pl.LightningModule.__init__(self)
        try:
            self.pre_concept_model = c_extractor_arch(
                pretrained=pretrain_model
            )
            
            if existing_weights != '':
                self.pre_concept_model.load_state_dict(torch.load(existing_weights))
            
        except:
            self.pre_concept_model = c_extractor_arch(output_dim=n_concepts)
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts) * (
                -5.0 if not sigmoidal_prob else 0.0
            )
        self.concat_prob = concat_prob
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        self.intervention_idxs = intervention_idxs
        self.related_concepts = related_concepts
        self.limit = limit
        self.adversarial_intervention = adversarial_intervention
        self.vae_model = vae_model
        self.stratified_model = stratified_model
        for i in range(n_concepts):
            if embeding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        # Two as each concept will have a positive and a
                        # negative embedding portion which are later mixed
                        2 * emb_size,
                    )
                )
            elif embeding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embeding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
                
                
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                if self.stratified_model:
                    # Use only the first half of each embedding to get probabilities 
                    self.concept_prob_generators.append(
                        torch.nn.Linear(
                            emb_size,
                            1,
                        )
                    )
                else:
                    # Then we will use one and only one probability generator which
                    # will be shared among all concepts. This will force concept
                    # embedding vectors to be pushed into the same latent space
                    self.concept_prob_generators.append(
                        torch.nn.Linear(
                            2 * emb_size,
                            1,
                        )
                    )
            elif not self.shared_prob_gen:
                if self.stratified_model:
                    self.concept_prob_generators.append(
                        torch.nn.Linear(
                            emb_size,
                            1,
                        )
                    )
                else:
                    self.concept_prob_generators.append(
                        torch.nn.Linear(
                            2 * emb_size,
                            1,
                        )
                    )
                
        if self.stratified_model:
            # Go from the emb_size/2 to n_tasks 
            self.c2y_model = torch.nn.Sequential(*[
                torch.nn.Linear(emb_size//2+int(concat_prob),n_tasks)])
        else:
            self.c2y_model = torch.nn.Sequential(*[
                torch.nn.Linear(
                    n_concepts * (emb_size + int(concat_prob)),
                    n_tasks,
                ),
            ])
        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss()
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss()
        )
        self.concept_loss_weight = concept_loss_weight
        self.concept_pair_loss_weight = concept_pair_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_embedding = sigmoidal_embedding
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        
        self.all_active_concepts = [np.zeros((0,self.emb_size)) for i in range(self.n_concepts)]
        self.all_inactive_concepts = [np.zeros((0,self.emb_size)) for i in range(self.n_concepts)]
        self.concept_folder_location = "cem_concepts/{}/{}".format(experiment_name,seed)
        
        if not os.path.exists(self.concept_folder_location):
            os.makedirs(self.concept_folder_location)
        
        self.experiment_name = experiment_name
                                
    def write_concepts(self):
        for concept_num in range(self.n_concepts):
            active_concept_file = self.concept_folder_location+"/{}_concept_{}_active.npy".format(self.experiment_name,concept_num)
            inactive_concept_file = self.concept_folder_location+"/{}_concept_{}_inactive.npy".format(self.experiment_name,concept_num)

            np.save(open(active_concept_file,"wb"),self.all_active_concepts[concept_num])
            np.save(open(inactive_concept_file,"wb"),self.all_inactive_concepts[concept_num])
            
    def _after_interventions(
        self,
        prob,
        concept_idx,
        intervention_idxs=None,
        c_true=None,
        train=False,
    ):
                
        if train and (self.training_intervention_prob != 0) and (
            intervention_idxs is None
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs and self.related_concepts == None:
            return prob
        # Ours: Intervene with related concepts
        elif concept_idx not in intervention_idxs and self.related_concepts != None and concept_idx in self.related_concepts:
            if any(key in intervention_idxs for key in self.related_concepts[concept_idx].keys()):
                full_prob = 0
                num_tot = 0
                
                potential_concepts = [i for i in self.related_concepts[concept_idx] if i in intervention_idxs]
                random.shuffle(potential_concepts)
                
                if self.limit != None and self.limit > 0:
                    potential_concepts = potential_concepts[:self.limit]
                
                for reference_concept in potential_concepts: 
                    function, confidence = self.related_concepts[concept_idx][reference_concept]
                    true_concept_value = function(c_true[:,reference_concept:reference_concept+1])
                    full_prob +=  (
                        (
                           true_concept_value *
                            self.active_intervention_values[concept_idx]
                        ) +
                        (
                            (true_concept_value  - 1) *
                            -self.inactive_intervention_values[concept_idx]
                        )
                    )

                    num_tot += 1


                full_prob /= len(potential_concepts)
                prob = full_prob * confidence + (1-confidence)*prob
            

            return prob
            
        c_true = self._switch_concepts(c_true)
        if self.sigmoidal_prob:
            return c_true[:, concept_idx:concept_idx+1]
        result = (
            (
                c_true[:, concept_idx:concept_idx+1] *
                self.active_intervention_values[concept_idx]
            ) +
            (
                (c_true[:, concept_idx:concept_idx+1] - 1) *
                -self.inactive_intervention_values[concept_idx]
            )
        )
        return result
    
    def _forward(self, x, intervention_idxs=None, c=None, train=False):
        # Run Resnet over the dataset
        pre_c = self.pre_concept_model(x)
        probs = []
        full_vectors = []
        sem_probs = []
        
        all_contexts = []

        zero_shot = False 

        if zero_shot:
            all_probs = torch.stack([self.sig(self.concept_prob_generators[0](self.sig(context_gen(pre_c))))[:,0] for i, context_gen in enumerate(self.concept_context_generators)])
            
            certainty = all_probs * (1-all_probs)
            certainty = torch.mean(certainty,axis=1)
            zero_shot_intervention = sorted(list(torch.topk(certainty,k=20,largest=False)[1].detach().tolist()))
            rounded_c = torch.round(all_probs).T


        for i, context_gen in enumerate(self.concept_context_generators):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
                
            # Get concepts from there using the concept generator
            context = context_gen(pre_c)
            
            if self.sigmoidal_embedding:
                context = self.sig(context)
                
            # From the concept get the probability
            prob = prob_gen(context)
            sem_probs.append(self.sig(prob))
            if self.sigmoidal_prob:
                prob = self.sig(prob)
            prob = self._after_interventions(
                prob,
                concept_idx=i,
                intervention_idxs=intervention_idxs,
                c_true=c,
                train=train,
            )

            # # TODO: Remove this, just to test out one-shot
            # if (intervention_idxs == None or intervention_idxs == []) and zero_shot:
            #     prob = self._after_interventions(
            #         prob,
            #         concept_idx=i,
            #         intervention_idxs=zero_shot_intervention,
            #         c_true=rounded_c,
            #         train=train,
            #     )

            probs.append(prob)
            # Then time to mix!
            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            
            all_contexts.append(context)
            
            # TODO: Make sure that we analyze only the non-probability part of the embedding 
            # Write which concepts were used to compute outputs
            if not train and c != None:
                self.update_concepts(context,self.emb_size,c,i)
            else:
                # Reset concepts for the next run
                self.reset_concepts()
            
            mask = prob if self.sigmoidal_prob else self.sig(prob)
            context = context_pos * mask + context_neg * (1 - mask)
                
    
            if self.concat_prob:
                # Then the probability bit will be added
                # as part of the bottleneck
                full_vectors.append(torch.cat(
                    [context, probs[i]],
                    axis=-1,
                ))
            else:
                # Otherwise, let's completely ignore the probability bit
                full_vectors.append(context)
                
        c_sem = torch.cat(sem_probs, axis=-1)
        c_pred = torch.cat(full_vectors, axis=-1)
        
        y = self.c2y_model(c_pred)
        contexts = torch.stack(all_contexts)
                
        return c_sem, c_pred, y, contexts

    def _run_step(self, batch, batch_idx, train=False):
        x, y, c = self._unpack_batch(batch)
        y = y.long()
        if self.intervention_idxs is not None:
            if self.related_concepts is not None or self.vae_model is not None:
                c_sem, c_logits, y_logits,context = self._forward(
                    x,
                    intervention_idxs=self.intervention_idxs,
                    c=c,
                    train=train,
                )

            else:
                c_sem, c_logits, y_logits,context = self._forward(
                    x,
                    intervention_idxs=self.intervention_idxs,
                    c=c,
                    train=train,
                )
            
        else:
            c_sem, c_logits, y_logits,context = self._forward(x, c=c, train=train)
            
        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0
        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            concept_loss = self.loss_concept(c_sem, c)
            loss = self.concept_loss_weight * concept_loss + task_loss
            concept_loss_scalar = concept_loss.detach()
        else:
            loss = task_loss
            concept_loss_scalar = 0.0
            
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1] + concept_pair_loss_weight * c.shape[-1])
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            top_k_val = self.top_k_accuracy
            if len(labels) == 2:
                y_pred = y_pred[:,0]
                top_k_val = 1
            y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                y_true,
                y_pred,
                k=top_k_val,
                labels=labels,
            )
            result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result
    
    def reset_concepts(self):
        # Save our current concepts before writing 
        if len(self.all_active_concepts[0])>0:
            self.write_concepts()
        
        self.all_active_concepts = [np.zeros((0,self.emb_size)) for i in range(self.n_concepts)]
        self.all_inactive_concepts = [np.zeros((0,self.emb_size)) for i in range(self.n_concepts)]
                    
    def update_concepts(self,concept_vectors,embedding_size,concept_present,concept_num): 
        active_concepts = concept_vectors[concept_present[:,concept_num] == True][:,:embedding_size].cpu().detach().numpy()
        inactive_concepts = concept_vectors[concept_present[:,concept_num] == False][:,embedding_size:].cpu().detach().numpy()
                
        self.all_active_concepts[concept_num] = np.concatenate((self.all_active_concepts[concept_num],active_concepts))
        self.all_inactive_concepts[concept_num] = np.concatenate((self.all_inactive_concepts[concept_num],inactive_concepts))
