import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
import tensorflow as tf

def create_skipgram_architecture(embedding_dimension,vocab_size,initial_embedding=None):
    """Create a skipgram architecture in keras, given an embedding dimension and a vocab size
    Do this by first turning a (target,context) pair into embeddings, taking the dot, then 
        enforcing that they're close to 1
    
    Arguments:
        embedding_dimension: Number representing the size of each embedding vector, such as 128
        vocab_size: Total number of words in the vocabulary; for concepts, total number of concepts
        
    Returns: Keras Model
    """
    
    # Target
    w_inputs = Input(shape=(1, ), dtype='int32')
    
    if type(initial_embedding) != type(None):
        w = Embedding(vocab_size, embedding_dimension, 
                     embeddings_initializer=tf.keras.initializers.Constant(initial_embedding))(w_inputs)
    else: 
        w = Embedding(vocab_size, embedding_dimension)(w_inputs)

    # Context 
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(vocab_size, embedding_dimension)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
    
    return SkipGram
