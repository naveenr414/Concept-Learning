from src.metrics import *
from src.hierarchy import *
from src.concept_vectors import *

def test_stability():
    mnist_attributes = get_mnist_attributes()
    
    assert stability_metric(create_linkage_hierarchy,
                            load_cem_vectors_simple,'mnist',mnist_attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,
                            load_tcav_vectors_simple,'mnist',mnist_attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,
                            load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45])>0
    assert stability_metric(create_linkage_hierarchy,
                            load_cem_vectors_simple,'mnist',mnist_attributes,[43,44,45])>0

def test_image_robustness():
    mnist_attributes = get_mnist_attributes()
    
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,'mnist',mnist_attributes,[43]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,'mnist',mnist_attributes,[44]) > 0
    
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_label_vectors_simple,'mnist',mnist_attributes,[43,44,45]) == 0
    
def test_image_responsiveness():
    mnist_attributes = get_mnist_attributes()
    
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,'mnist',mnist_attributes,[43,44,45]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,'mnist',mnist_attributes,[43]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,'mnist',mnist_attributes,[44])>0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_label_vectors_simple,'mnist',mnist_attributes,[43,44,45]) > 20
    
if __name__ == "__main__":
    test_stability()
    test_image_robustness()
    test_image_responsiveness()