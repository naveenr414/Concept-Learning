from src.metrics import *
from src.hierarchy import *
from src.concept_vectors import *
from src.dataset import *

def test_stability():
    dataset = MNIST_Dataset()
    attributes = dataset.get_attributes()
    
    assert stability_metric(create_linkage_hierarchy,
                            load_cem_vectors_simple,dataset,attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,
                            load_tcav_vectors_simple,dataset,attributes,[45,45]) == 0
    assert stability_metric(create_linkage_hierarchy,
                            load_tcav_vectors_simple,dataset,attributes,[43,44,45])>0
    assert stability_metric(create_linkage_hierarchy,
                            load_cem_vectors_simple,dataset,attributes,[43,44,45])>0

def test_image_robustness():
    dataset = MNIST_Dataset()
    attributes = dataset.get_attributes()
    
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,dataset,attributes,[43,44,45]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,dataset,attributes,[43]) > 0
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,dataset,attributes,[44]) > 0
    
    assert robustness_image_metric(create_linkage_hierarchy,
                                   load_label_vectors_simple,dataset,attributes,[43,44,45]) == 0
    
def test_image_responsiveness():
    dataset = MNIST_Dataset()
    attributes = dataset.get_attributes()
    
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,dataset,attributes,[43,44,45]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,dataset,attributes,[43]) > 0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,dataset,attributes,[44])>0
    assert responsiveness_image_metric(create_linkage_hierarchy,
                                       load_label_vectors_simple,dataset,attributes,[43,44,45]) > 20
    
def test_truthfulness():
    dataset = MNIST_Dataset()
    attributes = dataset.get_attributes()

    assert 0<truthfulness_image_metric(create_linkage_hierarchy,
                                       load_tcav_vectors_simple,dataset,attributes,[43,44,45])<1
    assert 0<truthfulness_image_metric(create_linkage_hierarchy,
                                   load_tcav_vectors_simple,dataset,attributes,[43])<1
    assert 0<truthfulness_image_metric(create_linkage_hierarchy,
                                   load_label_vectors_simple,dataset,attributes,[43])<1
    
    
if __name__ == "__main__":
    test_stability()
    test_image_robustness()
    test_image_responsiveness()
    test_truthfulness()