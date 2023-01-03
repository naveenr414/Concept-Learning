from src.hierarchy import *
import numpy as np

def test_hierarchy_class():
    h = Hierarchy(None)
    
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
