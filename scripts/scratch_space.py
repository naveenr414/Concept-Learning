# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook

import os
os.chdir('../')
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import matplotlib.pyplot as plt
import random
from src.dataset import *
from src.concept_vectors import *
from src.util import *
from src.plots import *
from src.hierarchy import *
from src.metrics import *
from src.models import * 
from src.create_vectors import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
import pickle
import sklearn
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json


# ## OIS

dataset = MNIST_Dataset()
attributes = dataset.get_attributes() 
suffix = "" 
seed = 43


def concept_purity_matrix(
    c_soft,
    c_true,
    concept_label_cardinality=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    ignore_diags=False,
    jointly_learnt=False,
):
    """
    Computes a concept purity matrix where the (i,j)-th entry represents the
    predictive accuracy of a classifier trained to use the i-th concept's soft
    labels (as given by c_soft_train) to predict the ground truth value of the
    j-th concept.

    This process is informally defined only for binary concepts by Mahinpei et
    al.'s in "Promises and Pitfalls of Black-Box Concept Learning Models".
    Nevertheless, this method supports both binary concepts (given as a 2D
    matrix in c_soft) or categorical concepts (given by a list of 2D matrices
    in argument c_soft).

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param List[int] concept_label_cardinality: If given, then this is a list
        of integers such that its i-th index contains the number of classes
        that the it-th concept may take. If not given, then we will assume that
        all concepts have the same cardinality as the number of activations in
        their soft representations.
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept soft representations to predict the j-th concept.
    """
    # Start by handling default arguments
    predictor_train_kwags = predictor_train_kwags or {}

    # Check that their rank is the expected one
    assert len(c_true.shape) == 2, (
        f'Expected testing concept predictions to be a matrix with shape '
        f'(n_samples, n_concepts) but instead got a matrix with shape '
        f'{c_true.shape}'
    )

    # Construct a list concept_label_cardinality that maps a concept to the
    # cardinality of its label set as specified by the testing data
    (n_samples, n_true_concepts) = c_true.shape
    if isinstance(c_soft, np.ndarray):
        n_soft_concepts = c_soft.shape[-1]
    else:
        assert isinstance(c_soft, list), (
            f'c_soft must be passed as either a list or a np.ndarray. '
            f'Instead we got an instance of "{type(c_soft).__name__}".'
        )
        n_soft_concepts = len(c_soft)

    assert n_soft_concepts >= n_true_concepts, (
        f'Expected at least as many soft concept representations as true '
        f'concepts labels. However we received {n_soft_concepts} soft concept '
        f'representations per sample while we have {n_true_concepts} true '
        f'concept labels per sample.'
    )

    if isinstance(c_soft, np.ndarray):
        # Then, all concepts must have the same representation size
        assert c_soft.shape[0] == c_true.shape[0], (
            f'Expected a many test soft-concepts as ground truth test '
            f'concepts. Instead got {c_soft.shape[0]} soft-concepts '
            f'and {c_true.shape[0]} ground truth test concepts.'
        )
        if concept_label_cardinality is None:
            concept_label_cardinality = [2 for _ in range(n_soft_concepts)]
        # And for simplicity and consistency, we will rewrite c_soft as a
        # list such that i-th entry contains an array with shape
        # (n_samples, repr_size) indicating the representation of the i-th
        # concept for all samples
        new_c_soft = [None for _ in range(n_soft_concepts)]
        for i in range(n_soft_concepts):
            if len(c_soft.shape) == 1:
                # If it is a scalar representation, then let's make it explicit
                new_c_soft[i] = np.expand_dims(c_soft[..., i], axis=-1)
            else:
                new_c_soft[i] = c_soft[..., i]
        c_soft = new_c_soft
    else:
        # Else, time to infer these values from the given list of soft
        # labels
        assert isinstance(c_soft, list), (
            f'c_soft must be passed as either a list or a np.ndarray. '
            f'Instead we got an instance of "{type(c_soft).__name__}".'
        )
        if concept_label_cardinality is None:
            concept_label_cardinality = [None for _ in range(n_soft_concepts)]
            for i, soft_labels in enumerate(c_soft):
                concept_label_cardinality[i] = max(soft_labels.shape[-1], 2)
                assert soft_labels.shape[0] == c_true.shape[0], (
                    f"For concept {i}'s soft labels, we expected "
                    f"{c_true.shape[0]} samples as we were given that many "
                    f"in the ground-truth array. Instead we found "
                    f"{soft_labels.shape[0]} samples."
                )

    # Handle the default parameters for both the generating function and
    # the concept label cardinality
    if predictor_model_fn is None:
        # Then by default we will use a simple MLP classifier with one hidden
        # ReLU layer with 32 units in it
        def predictor_model_fn(
            output_concept_classes=2,
        ):
            estimator = tf.keras.models.Sequential([
                tf.keras.layers.Dense(
                    32,
                    activation='relu',
                    name="predictor_fc_1",
                ),
                tf.keras.layers.Dense(
                    output_concept_classes if output_concept_classes > 2 else 1,
                    # We will merge the activation into the loss for numerical
                    # stability
                    activation=None,
                    name="predictor_fc_out",
                ),
            ])
            if jointly_learnt:
                loss = tf.nn.sigmoid_cross_entropy_with_logits
            else:
                loss = (
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ) if output_concept_classes > 2 else
                    tf.keras.losses.BinaryCrossentropy(
                        from_logits=True,
                    )
                )
            estimator.compile(
                # Use ADAM optimizer by default
                optimizer='adam',
                # Note: we assume labels come without a one-hot-encoding in the
                #       case when the concepts are categorical.
                loss=loss,
            )
            return estimator

    predictor_train_kwags = predictor_train_kwags or {
        'epochs': 25,
        'batch_size': min(512, n_samples),
        'verbose': 0,
    }

    # Time to start formulating our resulting matrix
    result = np.zeros((n_soft_concepts, n_true_concepts), dtype=np.float32)

    # Split our test data into two subsets as we will need to train
    # a classifier and then use that trained classifier in the remainder of the
    # data for computing our scores
    train_indexes, test_indexes = train_test_split(
        list(range(n_samples)),
        test_size=test_size,
    )

    for src_soft_concept in tqdm(range(n_soft_concepts)):

        # Construct a test and training set of features for this concept
        concept_soft_train_x = c_soft[src_soft_concept][train_indexes, ...]
        concept_soft_test_x = c_soft[src_soft_concept][test_indexes, ...]
        if len(concept_soft_train_x.shape) == 1:
            concept_soft_train_x = tf.expand_dims(
                concept_soft_train_x,
                axis=-1,
            )
            concept_soft_test_x = tf.expand_dims(
                concept_soft_test_x,
                axis=-1,
            )
        if jointly_learnt:
            # Construct a new estimator for performing this prediction
            output_size = 0
            for tgt_true_concept in range(n_true_concepts):
                output_size += (
                    concept_label_cardinality[tgt_true_concept]
                    if concept_label_cardinality[tgt_true_concept] > 2
                    else 1
                )
            estimator = predictor_model_fn(output_size)
            # Train it
            estimator.fit(
                concept_soft_train_x,
                c_true[train_indexes, :],
                **predictor_train_kwags,
            )
            # Compute the AUC of this classifier on the test data
            preds = estimator.predict(concept_soft_test_x)
            for tgt_true_concept in range(n_true_concepts):
                true_concepts = c_true[test_indexes, tgt_true_concept]
                used_preds = preds[:, tgt_true_concept]
                if concept_label_cardinality[tgt_true_concept] > 2:
                    # Then lets apply a softmax activation over all the probability
                    # classes
                    used_preds = scipy.special.softmax(used_preds, axis=-1)

                    # And make sure we only compute the AUC of labels that are
                    # actually used
                    used_labels = np.sort(np.unique(true_concepts))

                    # And select just the labels that are in fact being used
                    true_concepts = tf.keras.utils.to_categorical(
                        true_concepts,
                        num_classes=concept_label_cardinality[tgt_true_concept],
                    )[:, used_labels]
                    used_preds = used_preds[:, used_labels]
                if len(np.unique(true_concepts)) > 1:
                    auc = sklearn.metrics.roc_auc_score(
                        true_concepts,
                        used_preds,
                        multi_class='ovo',
                    )
                else:
                    if concept_label_cardinality[tgt_true_concept] <= 2:
                        used_preds = (
                            scipy.special.expit(used_preds) >= 0.5
                        ).astype(np.int32)
                    else:
                        used_preds = np.argmax(used_preds, axis=-1)
                        true_concepts = np.argmax(true_concepts, axis=-1)
                    auc = sklearn.metrics.accuracy_score(
                        true_concepts,
                        used_preds,
                    )

                # Finally, time to populate the actual entry of our resulting
                # matrix
                result[src_soft_concept, tgt_true_concept] = auc
        else:
            for tgt_true_concept in range(n_true_concepts):
                # Let's populate the (i,j)-th entry of our matrix by first
                # training a classifier to predict the ground truth value of
                # concept j using the soft-concept labels for concept i.
                if ignore_diags and (src_soft_concept == tgt_true_concept):
                    # Then for simplicity sake we will simply set this to one
                    # as it is expected to be perfectly predictable
                    result[src_soft_concept, tgt_true_concept] = 1
                    continue

                # Construct a new estimator for performing this prediction
                estimator = predictor_model_fn(
                    concept_label_cardinality[tgt_true_concept]
                )
                # Train it
                estimator.fit(
                    concept_soft_train_x,
                    c_true[train_indexes, tgt_true_concept:(tgt_true_concept + 1)],
                    **predictor_train_kwags,
                )

                # Compute the AUC of this classifier on the test data
                preds = estimator.predict(concept_soft_test_x)
                true_concepts = c_true[test_indexes, tgt_true_concept]
                if concept_label_cardinality[tgt_true_concept] > 2:
                    # Then lets apply a softmax activation over all the
                    # probability classes
                    preds = scipy.special.softmax(preds, axis=-1)

                    # And make sure we only compute the AUC of labels that are
                    # actually used
                    used_labels = np.sort(np.unique(true_concepts))

                    # And select just the labels that are in fact being used
                    true_concepts = tf.keras.utils.to_categorical(
                        true_concepts,
                        num_classes=concept_label_cardinality[tgt_true_concept],
                    )[:, used_labels]
                    preds = preds[:, used_labels]

                auc = sklearn.metrics.roc_auc_score(
                    true_concepts,
                    preds,
                    multi_class='ovo',
                )

                # Finally, time to populate the actual entry of our resulting
                # matrix
                result[src_soft_concept, tgt_true_concept] = auc

    # And that's all folks
    return result



def normalize_impurity(impurity, n_concepts):
    return impurity / (n_concepts / 2)



# +
def oracle_purity_matrix(
    concepts,
    concept_label_cardinality=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    jointly_learnt=False,
):
    """
    Computes an oracle's concept purity matrix where the (i,j)-th entry
    represents the predictive accuracy of a classifier trained to use the i-th
    concept (ground truth) to predict the ground truth value of the j-th
    concept.

    :param np.ndarray concepts: Ground truth concept values. Shape must be
        (n_samples, n_concepts).
    :param List[int] concept_label_cardinality: If given, then this is a list
        of integers such that its i-th index contains the number of classes
        that the it-th concept may take. If not given, then we will assume that
        all concepts are binary (i.e., concept_label_cardinality[i] = 2 for all
        i).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept label to predict the j-th concept.
    """

    return concept_purity_matrix(
        c_soft=concepts,
        c_true=concepts,
        concept_label_cardinality=concept_label_cardinality,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
        ignore_diags=True,
        jointly_learnt=jointly_learnt,
    )


# -

def oracle_impurity_score(
    c_soft,
    c_true,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    norm_fn=lambda x: np.linalg.norm(x, ord='fro'),
    oracle_matrix=None,
    purity_matrix=None,
    output_matrices=False,
    alignment_function=None,
    concept_label_cardinality=None,
    jointly_learnt=False,
    include_diagonal=True,
):
    """
    Returns the oracle impurity score (OIS) of the given soft concept
    representations `c_soft` with respect to their corresponding ground truth
    concepts `c_true`. This value is higher if concepts encode unnecessary
    information from other concepts in their soft representation and lower
    otherwise. If zero, then all soft concept labels are considered to be
    "pure".

    We compute this metric by calculating the norm of the absolute difference
    between the purity matrix derived from the soft concepts and the purity
    matrix derived from an oracle model. This oracle model is trained using
    the ground truth labels instead of the soft labels and may capture trivial
    relationships between different concept labels.

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.
    :param Function[(np.ndarray), float] norm_fn: A norm function applicable to
        a 2D numpy matrix representing the absolute difference between the
        oracle purity score matrix and the predicted purity score matrix. If not
        given then we will use the 2D Frobenius norm.
    :param np.ndarray oracle_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of an oracle that predicts the value of concept j given the
        ground truth of concept i. If not given, then this matrix will be
        computed using the ground truth concept labels.
    :param np.ndarray purity_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of predicting the value of concept j given the soft
        representation of concept i. If not given, then this matrix will be
        computed using the purity scores from the input soft representations.
    :param bool output_matrices: If True then this method will output a tuple
        (score, purity_matrix, oracle_matrix) containing the computed purity
        score, purity matrix, and oracle matrix given this function's
        arguments.
    :param Function[(np.ndarray), np.ndarray] alignment_function: an optional
        alignment function that takes as an input an (k, n_concepts) purity
        matrix, where k >= n_concepts and its (i, j) value is the AUC of
        predicting true concept j using soft representations i, and returns a
        (n_concepts, n_concepts) matrix where a subset of n_concepts soft
        concept representations has been aligned in a bijective fashion with
        the set of all ground truth concepts.


    :returns Or[Tuple[float, np.ndarray, np.ndarray], float]: If output_matrices
        is False (default behavior) then the output will be a non-negative float
        in [0, 1] representing the degree to which individual concepts
        representations encode unnecessary information for other concepts. Higher
        values mean more impurity and the concepts are considered to be pure if
        the returned value is 0. If output_matrices is True, then the output
        will be a tuple (score, purity_matrix, oracle_matrix) containing the
        computed purity score, purity matrix, and oracle matrix given this
        function's arguments. If alignment_function is given, then the purity
        matrix will be a tuple (purity_matrix, aligned_purity_matrix) containing
        the pre and post alignment purity matrices, respectively.
    """

    # Now the concept_label_cardinality vector from the given soft labels
    (n_samples, n_concepts) = c_true.shape
    if concept_label_cardinality is None:
        concept_label_cardinality = [
            len(set(c_true[:, i]))
            for i in range(n_concepts)
        ]
    # First compute the predictor soft-concept purity matrix
    if purity_matrix is not None:
        pred_matrix = purity_matrix
    else:
        pred_matrix = concept_purity_matrix(
            c_soft=c_soft,
            c_true=c_true,
            predictor_model_fn=predictor_model_fn,
            predictor_train_kwags=predictor_train_kwags,
            test_size=test_size,
            concept_label_cardinality=concept_label_cardinality,
            jointly_learnt=jointly_learnt,
        )

    # Compute the oracle's purity matrix
    if oracle_matrix is None:
        oracle_matrix = oracle_purity_matrix(
            concepts=c_true,
            concept_label_cardinality=concept_label_cardinality,
            predictor_model_fn=predictor_model_fn,
            predictor_train_kwags=predictor_train_kwags,
            test_size=test_size,
            jointly_learnt=jointly_learnt,
        )
    
    # Finally, compute the norm of the absolute difference between the two
    # matrices
    if alignment_function is not None:
        # Then lets make sure we align our prediction matrix correctly
        aligned_matrix = alignment_function(pred_matrix)
        if not include_diagonal:
            used_aligned_matrix = np.copy(aligned_matrix)
            np.fill_diagonal(used_aligned_matrix, 1)
            used_oracle_matrix = np.copy(oracle_matrix)
            np.fill_diagonal(used_oracle_matrix, 1)
        else:
            used_oracle_matrix = oracle_matrix
            used_aligned_matrix = aligned_matrix
        score = norm_fn(np.abs(used_oracle_matrix - used_aligned_matrix))
        if output_matrices:
            return score, (pred_matrix, aligned_matrix), oracle_matrix
        return score

    if not include_diagonal:
        used_pred_matrix = np.copy(pred_matrix)
        np.fill_diagonal(used_pred_matrix, 1)
        used_oracle_matrix = np.copy(oracle_matrix)
        np.fill_diagonal(used_oracle_matrix, 1)
    else:
        used_oracle_matrix = oracle_matrix
        used_pred_matrix = pred_matrix

    score = normalize_impurity(
        impurity=norm_fn(np.abs(used_oracle_matrix - used_pred_matrix)),
        n_concepts=n_concepts,
    )
    if output_matrices:
        return score, pred_matrix, oracle_matrix
    return score



def load_cem_vectors_ordered(experiment_name,seed):
    pass 


dataset = MNIST_Dataset()
c_pred_label = np.array([create_vector_from_label_valid(attribute,dataset,suffix,seed=-1)[0] for attribute in attributes]).T
groundtruth = c_pred_label

c_true = groundtruth.astype(float)
c_soft = c_pred_label.astype(float)

output = oracle_impurity_score(c_pred_label.astype(float),groundtruth.astype(float),predictor_train_kwags={
        'epochs': 300, # Defaults to 25
        'batch_size': 2024, # Defaults to 512 but change it if the dataset is not large enough
        'verbose': 0,
    },
    jointly_learnt=True,
    output_matrices=True,)

oracle_impurity_score(c_pred_label.astype(float),groundtruth.astype(float),jointly_learnt=True)

for experiment_name in ["mnist","cub","dsprites","chexpert"]:
    ois_scores = {}
    suffix = ""
    print("Experiment {}".format(experiment_name))
    for seed in [43,44,45]:
        ois_scores[seed] = {}
        if experiment_name == "mnist":
            dataset = MNIST_Dataset()
        elif experiment_name == "cub":
            dataset = CUB_Dataset()
        elif experiment_name == "dsprites":
            dataset = DSprites_Dataset()
        elif experiment_name == "chexpert":
            dataset = Chexpert_Dataset()
        else:
            raise Exception("Dataset {} not found".format(experiment_name))

        attributes = dataset.get_attributes()
        num_concepts = len(dataset.get_attributes())
        vectors_active = [np.load("results/bases/cem/{}/{}/{}_concept_{}_active.npy".format(experiment_name,seed,experiment_name,i)) for i in range(num_concepts)]
        vectors_inactive = [np.load("results/bases/cem/{}/{}/{}_concept_{}_inactive.npy".format(experiment_name,seed,experiment_name,i)) for i in range(num_concepts)]
        data = dataset.get_data(seed,train=False)

        all_cem_data = np.zeros((len(data),16,num_concepts))

        for j in range(num_concepts):
            active_counter, inactive_counter = 0,0 
            for i in range(len(data)):
                if data[i]['attribute_label'][j] == 0:
                    all_cem_data[i,:,j] = vectors_inactive[j][inactive_counter]
                    inactive_counter += 1
                else:
                    all_cem_data[i,:,j] = vectors_active[j][active_counter]
                    active_counter += 1

        c_pred_label = np.array([create_vector_from_label_valid(attribute,dataset,suffix,seed=-1)[0] for attribute in attributes]).T
        groundtruth = c_pred_label

        cem_score = oracle_impurity_score(all_cem_data,groundtruth.astype(float),predictor_train_kwags={
                'epochs': 300, # Defaults to 25
                'batch_size': 2024, # Defaults to 512 but change it if the dataset is not large enough
                'verbose': 0,
            },
            jointly_learnt=True,
            output_matrices=True,)[0]
        label_score = oracle_impurity_score(c_pred_label.astype(float),groundtruth.astype(float),predictor_train_kwags={
            'epochs': 300, # Defaults to 25
            'batch_size': 2024, # Defaults to 512 but change it if the dataset is not large enough
            'verbose': 0,
        },
        jointly_learnt=True,
        output_matrices=True,)[0]
        
        ois_scores[seed]['label'] = label_score 
        ois_scores[seed]['cem'] = cem_score 
    json.dump(ois_scores,open("results/evaluation/ois/{}.json".format(experiment_name),"w"))


z = 1/0

# ## Other Things

dataset = DSprites_Dataset()
for method,name in zip ([load_label_vectors_simple,load_cem_vectors_simple,load_concept2vec_vectors_simple],["labels","cem","concept2vec"]):
    for seed in [43,44,45]:
        file_name = "{}_{}".format(name,seed)
        save_concept_vectors(method,dataset,seed,file_name)

# +
from src.dataset import CUB_Dataset
from src.metrics import compute_all_metrics
from src.concept_vectors import load_concept2vec_vectors_simple
dataset = CUB_Dataset()
attributes = dataset.get_attributes()
method = load_concept2vec_vectors_simple
seeds = [43,44,45]

results = compute_all_metrics(method,
                                    dataset,
                                    attributes,
                                    seeds)

# +
from src.dataset import CUB_Dataset
from src.concept_vectors import load_shapley_vectors_simple, fix_predictions

fix_predictions(load_shapley_vectors_simple,CUB_Dataset(),43,
                    "results/logits/cub/train_c.npy","results/logits/cub/valid_c.npy","results/logits/cub/test_c.npy",
                    "dataset/CUB/preprocessed/train.pkl","dataset/CUB/preprocessed/val.pkl","dataset/CUB/preprocessed/test.pkl",
                    "results/logits/cub/train_fixed.npy","results/logits/cub/valid_fixed.npy","results/logits/cub/test_fixed.npy")
# -

train = np.load(open("results/logits/cub/train_c.npy","rb"))
train_fixed = np.load(open("results/logits/cub/train_fixed.npy","rb"))
true_vals = np.array([i['attribute_label'] for i in pickle.load(open("dataset/CUB/preprocessed/train.pkl","rb"))])

train_fixed.shape

np.sum(np.round(train_fixed) == true_vals)/true_vals.size

dataset.create_robustness

dataset = DSprites_Dataset()

dataset.get_data()[-1]

dataset.create_gaussian()

dataset.create_junk()

load_label_vectors_simple(dataset.get_attributes()[0],dataset,"",seed=43)

len(glob.glob(dataset.all_files))

dataset.all_files

model_name = "VGG16"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)
else:
    print("No GPus")


export_concept_vector(load_tcav_vectors_simple,"tcav",dataset,[43,44,45])

# ## Train large Chexpert models

train_large_image_model(dataset,model_name,"",seed=42)

for seed in range(44,46):
    for suffix in ["","_image_robustness","_image_responsiveness"]:
        print(seed,suffix)
        train_large_image_model(dataset,model_name,suffix,seed=seed)

for seed in range(43,46):
    for suffix in ["","_image_robustness","_image_responsiveness"]:
        print(seed,suffix)
        create_shapley_vectors(dataset.get_attributes(),dataset,suffix,seed=seed,model_name=model_name)

# ## Setup the DSprites Dataset

npz_file = np.load(open("dataset/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz","rb"))

[i for i in npz_file]

dataset = DSprites_Dataset()

load_label_vectors_simple(dataset.get_attributes()[0],dataset,"",seed=43)

h = create_hierarchy(create_ward_hierarchy, load_label_vectors_simple,dataset,'',dataset.get_attributes(),43)

# +
attribute_labels = list(set([tuple(i['attribute_label']) for i in dataset.get_data()]))
atttribute_labels = [list(i) for i in attribute_labels]

for i in attribute_labels:
    temp = []
    for j in range(len(i)):
        if i[j]:
            temp.append(dataset.get_attributes()[j])
    print(temp[1:])
# -

dataset.create_gaussian()

dataset.create_junk()

orientations = generate_random_orientation(10)
write_dataset(orientations,npz_file,write_images=True)



import pickle
a = pickle.load(open("dataset/dsprites/preprocessed/test.pkl","rb"))

a[0]

for suffix in ["","_image_robustness","_image_responsiveness"]:
    for seed in [43,44,45]:
        create_concept2vec(dataset,suffix,seed=seed)

dataset.create_gaussian()

'" "'.join(dataset.get_attributes())

len(a)

len(set([i['class_label'] for i in a]))

rand_orientations = generate_random_orientation(10)

image_has_orientation(rand_orientations[0],npz_file['latents_classes'][0])

image_has_orientation([0,0,0,0,0,0],npz_file['latents_classes'][0])

image_has_orientation([0,0,0,0,0,0],npz_file['latents_classes'][-1])

image_has_orientation([0,2,5,3,1,1],npz_file['latents_classes'][-1])

matching_0 = get_matching_images([0,0,1,0,0,0],npz_file)

len(matching_0)



len([i for i in range(100000) if image_has_orientation(orientation,latents[i])])

len(matching_0)

npz_file['imgs'][0].shape

write_image(0,npz_file)

len(npz_file['latents_classes'])

# ## Create Shapley Vectors for Varying Levels of Noise

for noise_std in [100]:
    train_large_image_model(dataset,model_name,"",seed=43,noise_std=noise_std)

for noise_std in [25,50,100]:
    create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=43,noise_std=noise_std)

train_large_image_model(MNIST_Dataset(),model_name,"",seed=42)

for label_flip in [0.01,0.05,0.1]:
    create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=43,label_flip=label_flip)

create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=43)

# ## Exploration of the Truthfulness Metric

similarities = get_model_concept_similarities(dataset,model)

attributes = dataset.get_attributes()

shapley_hierarchy = create_hierarchy(create_ward_hierarchy, load_shapley_vectors_simple,dataset,'',attributes,43)

print(shapley_hierarchy)

attributes = np.array(attributes)

for i in range(5):
    rand_index = random.randint(0,111)
    five_largest = np.argsort(similarities[rand_index])[::-1][:5]
    print(attributes[rand_index],attributes[five_largest])

truthfulness_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

# ## Re-train large models

model_name = "vgg16"

train_large_image_model(dataset,model_name,"",seed=42)

a = dataset.get_data(suffix="")

a[0]['img_path']

dataset.fix_path(a[0]['img_path'],'_image_robustness')

train_large_image_model(dataset,model_name,"_image_robustness",seed=43)

for seed in range(45,46):
    for suffix in ["","_image_robustness","_image_responsiveness"]:
        print(seed,suffix)
        train_large_image_model(dataset,model_name,suffix,seed=seed)

model_name = "inceptionv3"

# train_large_image_model(dataset,model_name,"",seed=43)
train_large_image_model(dataset,model_name,"",seed=44)
train_large_image_model(dataset,model_name,"",seed=45)

for seed in [44,45]:
    for suffix in ["","_image_robustness","_image_responsiveness"]:
        print(seed,suffix)
        create_shapley_vectors(dataset.get_attributes(),dataset,suffix,seed=seed,model_name=model_name)

create_shapley_vectors(dataset.get_attributes(),dataset,"_image_robustness",seed=43,model_name="resnet50")

create_shapley_vectors(dataset.get_attributes(),dataset,"_image_robustness",seed=43,model_name="inceptionv3")

robustness_image_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

# ## Create Shapley Vectors

for seed in range(43,46):
    for suffix in ["","_image_robustness","_image_responsiveness"]:
        print(seed,suffix)
        create_shapley_vectors(dataset.get_attributes(),dataset,suffix,seed=seed)

truthfulness_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[42])

create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=43)

create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=44)

create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=45)

create_shapley_vectors(dataset.get_attributes(),dataset,"_image_robustness",seed=43)

create_shapley_vectors(dataset.get_attributes(),dataset,"_image_responsiveness",seed=43)

for embedding_method in [load_label_vectors_simple,load_vae_concept_vectors_simple,load_cem_vectors_simple,load_concept2vec_vectors_simple,load_cem_stratified_vectors_simple]:
    embedding_method(dataset.get_attributes()[0],dataset,"",seed=43).shape

create_shapley_vectors(dataset.get_attributes(),dataset,"",seed=45)

create_shapley_vectors(MNIST_Dataset().get_attributes(),MNIST_Dataset(),"",seed=43)

# ## Compute Truthfulness Scores

for embedding_method in [load_shapley_vectors_simple]:
    print(embedding_method)
    t = truthfulness_metric_shapley(embedding_method,dataset,dataset.get_attributes(),[43,44,45],model_name="VGG16")
    print("{} {}".format(embedding_method,t))

truthfulness_metric_shapley(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[42],model_name="VGG16")

truthfulness_metric_shapley(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43],model_name="VGG16")

truthfulness_metric_shapley(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[44],model_name="VGG16")

truthfulness_metric_shapley(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[45],model_name="VGG16")

truthfulness_metric_shapley(load_label_vectors_simple,dataset,dataset.get_attributes(),[43],model_name="VGG16")

truthfulness_metric_shapley(load_random_vectors_simple,dataset,dataset.get_attributes(),[42],model_name="VGG16")

# ## Evaluating Shapley Vectors across metrics

truthfulness_metric_shapley(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43,44,45])

for embedding_method in [load_label_vectors_simple,load_cem_vectors_simple,load_vae_vectors_simple,load_concept2vec_vectors_simple]:
    t = truthfulness_metric_shapley(embedding_method,dataset,dataset.get_attributes(),[43,44,45])
    print(embedding_method,t)

stability_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43,44,45])

robustness_image_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

responsiveness_image_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

# ## Exploration of Shapley Vectors <-> Truthfulness relationship

model_name = "VGG16"

# +
model = get_large_image_model(dataset,model_name)
model.load_weights("results/models/{}_models/{}_45.h5".format(model_name.lower(),dataset.experiment_name))

similarity_matrix = get_model_concept_similarities(dataset,model)
# -

attributes = dataset.get_attributes()

seed = 45

all_embeddings = [embedding_method(i,dataset,"",seed=44)[0] for i in dataset.get_attributes()]
all_embeddings = np.array(all_embeddings)
distances = cdist(all_embeddings,all_embeddings,metric='cosine')

attribute = "has_eye_color::black"
attribute_idx = dataset.get_attributes().index(attribute)

co_occuring_concepts = find_similar_conepts_shapley(attribute,dataset,similarity_matrix,5)

top_5_concepts = np.argsort(distances[attribute_idx])[1:1+5]
co_occuring_concepts_hierarchy = np.array(dataset.get_attributes())[top_5_concepts]

co_occuring_concepts

co_occuring_concepts_hierarchy

# +
selected_concepts = random.sample(attributes,k=50)
selected_concepts_indices = [attributes.index(i) for i in selected_concepts]

temp_truthfulness = []

for concept in selected_concepts:
    co_occuring_concepts = find_similar_conepts_shapley(concept,dataset,similarity_matrix,5)
    co_occuring_concepts_hierarchy = rank_distance_concepts(embedding_method,dataset,concept,
                                                            co_occuring_concepts,seed)

    if len(co_occuring_concepts) == 1:
        temp_truthfulness.append(int(co_occuring_concepts == co_occuring_concepts_hierarchy))
    else:
        score = stats.kendalltau(co_occuring_concepts,
                                                  co_occuring_concepts_hierarchy).correlation
        
        if score < 0:
            print(concept,score)
            print(co_occuring_concepts)
            print(co_occuring_concepts_hierarchy)
        
        temp_truthfulness.append(score)
np.mean(temp_truthfulness)
# -



robustness_image_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

responsiveness_image_metric(load_shapley_vectors_simple,dataset,dataset.get_attributes(),[43])

shapley_hierarchy_mnist = create_hierarchy(create_ward_hierarchy, load_shapley_vectors_simple,MNIST_Dataset(),'',MNIST_Dataset().get_attributes(),43)

print(shapley_hierarchy_mnist)

shapley_hierarchy_cub_44 = create_hierarchy(create_ward_hierarchy, load_shapley_vectors_simple,CUB_Dataset(),'',CUB_Dataset().get_attributes(),44)

shapley_hierarchy_cub_45 = create_hierarchy(create_ward_hierarchy, load_shapley_vectors_simple,CUB_Dataset(),'',CUB_Dataset().get_attributes(),45)

print(shapley_hierarchy_cub_44)

print(shapley_hierarchy_cub_45)

shapley_robustness_hierarchy_cub_43 = create_hierarchy(create_ward_hierarchy, load_label_vectors_simple,CUB_Dataset(),'_image_robustness',CUB_Dataset().get_attributes(),43)

print(shapley_robustness_hierarchy_cub_43)

label_hierarchy_cub_43 = create_hierarchy(create_ward_hierarchy,load_shapley_vectors_simple,CUB_Dataset(),'',CUB_Dataset().get_attributes(),43)

print(label_hierarchy_cub_43)

normal_vector_distances = flat_distance_to_square(get_concept_distances(load_shapley_vectors_simple,dataset,"",dataset.get_attributes(),43))
normal_vector_distances_2 = flat_distance_to_square(get_concept_distances(load_shapley_vectors_simple,dataset,"",dataset.get_attributes(),44))
robust_vector_distances = flat_distance_to_square(get_concept_distances(load_shapley_vectors_simple,dataset,"_image_robustness",dataset.get_attributes(),43))

# +
k = 5

pairs_1 = get_top_k_pairs(normal_vector_distances,k=k)
pairs_2 = get_top_k_pairs(robust_vector_distances,k=k)
pairs_3 = get_top_k_pairs(normal_vector_distances_2,k=k)

intersection = set(pairs_1).intersection(set(pairs_2))
len(intersection)/(len(normal_vector_distances)*k)
# -

cem_hierarchy_cub = create_hierarchy(create_ward_hierarchy, load_cem_vectors_simple,CUB_Dataset(),'',CUB_Dataset().get_attributes(),43)

to_array = cem_hierarchy_cub.to_array(CUB_Dataset().get_attributes())

np.save(open("results/hierarchies/cem_cub.npy","wb"),to_array)

# ## Export Concept Vectors

for seed in [43,44,45]:
    for name, embedding_method in zip(["labels","shapley","cem","vae","concept2vec"],[load_label_vectors_simple,load_shapley_vectors_simple,
                                                            load_cem_vectors_simple,load_vae_vectors_simple,load_concept2vec_vectors_simple]):
        all_embeddings =  np.array([embedding_method(i,CUB_Dataset(),"",seed=seed)[0] for i in CUB_Dataset().get_attributes()])
        np.save(open("results/temp/{}_{}.npy".format(name,seed),"wb"),all_embeddings)

model = get_large_image_model(dataset,model_name)
model.load_weights("results/models/{}_models/{}_42.h5".format(model_name.lower(),dataset.experiment_name))

similarity_matrix = get_model_concept_similarities(dataset,model)
similarity_matrix.shape

shapley_vectors = np.array([load_shapley_vectors_simple(i,dataset,"",seed=42)[0] for i in dataset.get_attributes()])

shapley_vectors

similarity_matrix

# +
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)

num_classes = len(set([i['class_label'] for i in dataset.get_data()]))
num_attributes = len(dataset.get_data()[0]['attribute_label'])

data_valid = dataset.get_data(train=False)
img_paths_valid = ['dataset/'+i['img_path'] for i in data_valid]
labels_valid = [str(i['class_label']) for i in data_valid]
valid_df = pd.DataFrame(zip(img_paths_valid,labels_valid), columns=["image_path", "label"])

valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                          x_col="image_path",
                                          y_col="label",
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          class_mode="categorical",
                                          shuffle=False)

# -

predictions = model.predict(valid_generator)

concepts = np.array([i['attribute_label'] for i in data_valid])
contribution_array = np.array([[contribution_score(concepts,predictions,concept_num,class_num) for class_num in range(num_classes)]  for concept_num in range(num_attributes)])

contribution_array

contribution_score(concepts,predictions,1,0)

# +
data = dataset.get_data(train=False)
img_paths = ['dataset/'+i['img_path'] for i in data]
labels = [str(i['class_label']) for i in data]

num_attributes = len(dataset.get_attributes())
num_classes = len(set(labels))

concept_vectors = {}
model_name = "VGG16"
model = get_large_image_model(dataset,model_name)
model.load_weights("results/models/{}_models/{}_{}.h5".format(model_name.lower(),dataset.experiment_name+'',42))

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)

valid_df = pd.DataFrame(zip(img_paths,labels), columns=["image_path", "label"])

valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                          x_col="image_path",
                                          y_col="label",
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          class_mode="categorical",
                                          shuffle=False)

predictions = model.predict(valid_generator)

concepts = np.array([i['attribute_label'] for i in data[:len(predictions)]])
contribution_array = np.array([[contribution_score(concepts,predictions,concept_num,class_num) 
                                for class_num in range(num_classes)] for concept_num in range(num_attributes)])

# -

contribution_array

similarity_matrix = get_model_concept_similarities(dataset,model)

similarity_matrix

concept = dataset.get_attributes()[0]
compare_concepts = 5
all_concept_embeddings = np.array([
    load_shapley_vectors_simple(i,dataset,"",seed=42) for i in dataset.get_attributes()])

all_concept_embeddings.shape

co_occuring_concepts = find_similar_conepts_shapley(concept,dataset,similarity_matrix,compare_concepts)
co_occuring_concepts_hierarchy = rank_distance_concepts(all_concept_embeddings,dataset,concept,
                                                        dataset.get_attributes(),42)[1:1+compare_concepts]

co_occuring_concepts

# ## Create model

# +
num_attributes = len([i['attribute_label'] for i in dataset.get_data()][0])
if model_name.lower() == "vgg16":
    model_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
else:
    raise Exception("{} model not implemented yet".format(model_name))

for layer in model_base.layers:
    layer.trainable = False

fc1 = Flatten()(model_base.output)
fc1 = Dense(1024, activation='relu')(fc1)
output = Dense(num_attributes, activation='sigmoid')(fc1)

model = Model(inputs=model_base.input, outputs=output)

model.compile(loss='binary_crossentropy',
          optimizer=Adam(lr=0.0001),
          metrics=['accuracy'])
# -

data = dataset.get_data()

# +
img_paths = ['dataset/'+i['img_path'] for i in data]
attributes = [i['attribute_label'] for i in data]

data_dict = {'image_path': img_paths}
for i in range(num_attributes):
    data_dict['attribute' + str(i)] = [int(attribute[i]) for attribute in attributes]

df = pd.DataFrame(data_dict)

# Create an ImageDataGenerator with the desired data augmentation settings
datagen = ImageDataGenerator(
    rescale=1./255,)

batch_size = 32
image_size = (224, 224)

data = dataset.get_data(seed=43,suffix='')
generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='image_path',
    y_col=['attribute' + str(i) for i in range(num_attributes)],
    batch_size=batch_size,
    target_size=image_size,
    class_mode='other')

# -

model.fit(generator,
       steps_per_epoch=len(generator),
       epochs=10)

data = dataset.get_data()

img_paths = ['dataset/'+i['img_path'] for i in data]
attributes = np.array([i['attribute_label'] for i in data])
num_attributes = len(attributes[0])


# Define a function to load and preprocess the images
def load_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=input_shape[:2])
    img = tf.keras.utils.img_to_array(img) / 255.
    return img



input_shape = (224, 224, 3)


images = np.array([load_image(img_path) for img_path in img_paths])

# +
input_tensor = Input(shape=input_shape)

# Define the base model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add the new layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
# Define the output layer
output_layer = Dense(num_attributes, activation='sigmoid')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['mse'])

# -

epochs = 10
batch_size = 32
model.fit(images, attributes, batch_size=batch_size, epochs=epochs)

model(np.array([images[0]])) > 0.5

attributes[0]

model.save_weights("results/models/vgg16_models/cub_concept.h5")

model(images,batch_size=32)

logits = model.predict(images, batch_size=32)


