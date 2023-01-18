from src.metrics import *
from src.hierarchy import *
from src.concept_vectors import *

def test_stability():
    mnist_attributes = ['{}_color'.format(i) for i in range(5)] + ['{}_number'.format(i) for i in range(5)]
    
    assert stability_metric(create_linkage_hierarchy,load_cem_vectors_simple,'mnist',mnist_attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45])>0
    assert stability_metric(create_linkage_hierarchy,load_cem_vectors_simple,'mnist',mnist_attributes,[43,44,45])>0

def test_image_robustness():
    mnist_attributes = ['{}_color'.format(i) for i in range(5)] + ['{}_number'.format(i) for i in range(5)]
    
    assert robustness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[43]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[44]) > 0
    
    assert robustness_image_metric(create_linkage_hierarchy,load_label_vectors_simple,'mnist',mnist_attributes,[44,45]) == 0
    
def test_image_responsiveness():
    mnist_attributes = ['{}_color'.format(i) for i in range(5)] + ['{}_number'.format(i) for i in range(5)]
    
    assert responsiveness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[43]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,load_tcav_vectors_simple,'mnist',mnist_attributes,[44])>0
    
    assert responsiveness_image_metric(create_linkage_hierarchy,load_label_vectors_simple,'mnist',mnist_attributes,[44,45]) == 24
    
if __name__ == "__main__":
    test_stability()
    test_image_robustness()
    test_image_responsiveness()