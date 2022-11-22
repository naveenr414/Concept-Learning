from src.util import *
import numpy as np

def test_avg_distance():
    data = np.array([[0,0]])
    classes = [0]
    ret = find_average_distances(data,classes)
    assert ret.shape == (1,1)
    assert ret[0][0] == 0
    
    data = np.array([[0,0],[0,0]])
    classes = [0,1]
    ret = find_average_distances(data,classes)
    assert ret.shape == (2,2)
    assert ret[0][0] == ret[1][1] == 0
    assert ret[0][1] == ret[1][0] == 0
    
    data = np.array([[0,0],[1,0]])
    classes = [0,1]
    ret = find_average_distances(data,classes)
    assert ret.shape == (2,2)
    assert ret[0][0] == ret[1][1] == 0
    assert ret[0][1] == ret[1][0] == 1
    
    data = np.array([[0,0],[1,0],[1,0]])
    classes = [0,1,0]
    ret = find_average_distances(data,classes)
    assert ret.shape == (2,2)
    assert ret[1][1] == 0
    assert ret[0][0] == 1
    assert ret[0][1] == ret[1][0] == 0.5
    
def test_list_functions():
    assert encode_list(['cat','dog','dog','cat']) == [0,1,1,0]
    assert find_unique_in_order(['cat','dog','dog','cat']) == ['cat','dog']
    
if __name__ == '__main__':
    test_avg_distance()
    test_list_functions()
    print("All tests passed")