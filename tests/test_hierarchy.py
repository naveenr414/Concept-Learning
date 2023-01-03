from src.hierarchy import *
import numpy as np

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

if __name__ == "__main__":
    test_hierarchy_methods()
    test_hierarchy_class()
