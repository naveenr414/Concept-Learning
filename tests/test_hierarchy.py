from src.hierarchy import *
import numpy as np
from src.concept_vectors import *

def test_hierarchy_class():
    h = Hierarchy()
    
    random_vector_names = [str(i) for i in range(5)]
    random_vectors = np.random.random((5,6))
    sample_numpy_hierarchy = create_linkage_hierarchy(random_vectors)
    h.from_array(sample_numpy_hierarchy,random_vector_names)
    
    arr_difference = abs(h.to_array(random_vector_names)-sample_numpy_hierarchy)
        
    assert arr_difference.max()<.01
    assert h.root_split != None
    assert h.root_split.left_split != None
    assert h.root_split.right_split != None
    
    assert len(str(h)) > 0
    assert len(str(h).split("\n")) > 5
    
    np.random.seed(42)
    r1 = np.random.random((5,6))
    r2 = np.random.random((5,6))
    
    h1 = Hierarchy()
    h2 = Hierarchy()
    
    sample_numpy_hierarchy = create_linkage_hierarchy(r1)
    h1.from_array(sample_numpy_hierarchy,random_vector_names)
    
    sample_numpy_hierarchy = create_linkage_hierarchy(r2)
    h2.from_array(sample_numpy_hierarchy,random_vector_names)
    
    assert h1.distance(h2) == 8.0
    
def test_hierarchy_methods():
    random_vectors = np.random.random((5,6))
    
    h = create_linkage_hierarchy(random_vectors)
    assert len(h) > 0
    assert len(h) == len(random_vectors)-1
    assert h.shape[1] == 4
    
    h = create_ward_hierarchy(random_vectors)
    assert len(h) > 0
    assert len(h) == len(random_vectors)-1
    assert h.shape[1] == 4

def test_hierarchy_simplified():
    h1 = create_hierarchy(create_ward_hierarchy,load_cem_vectors_simple, 'xor',['0','1'],42)
    h2 = create_hierarchy(create_linkage_hierarchy,load_tcav_vectors_simple, 'unfiled',['soccer','alga'],42)
    h3 = create_hierarchy(create_linkage_hierarchy,load_label_vectors_simple, 'cub',["has_bill_shape::dagger",'has_bill_shape::hooked_seabird'],42)
        
    assert type(h1) == type(h2)
    assert type(h2) == type(h3)
    assert type(h1) == type(h3)
    assert h1 != None
    assert h2 != None
    assert h3 != None
    
    assert len(str(h1)) > 25
    assert len(str(h2)) > 25
    assert len(str(h3)) > 50
    
if __name__ == "__main__":
    test_hierarchy_simplified()
    #test_hierarchy_methods()
    #test_hierarchy_class()
