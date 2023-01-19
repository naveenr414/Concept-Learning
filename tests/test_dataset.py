from src.dataset import *

def test_cub():
    dataset = CUB_Dataset()
    
    all_attributes = dataset.get_attributes()
    assert len(all_attributes) == 112
    
    assert len(dataset.get_data()) > 0
    
    train = dataset.get_data()
    assert 'img_path' in train[0]
    assert 'attribute_label' in train[0]

    with_attribute = dataset.get_images_with_attribute(all_attributes[0])
    without_attribute = dataset.get_images_without_attribute(all_attributes[0])
    
    assert len(with_attribute)>0
    assert len(without_attribute)>0
    assert with_attribute[0] not in without_attribute
    assert without_attribute[0] not in with_attribute
    
def test_attribute_folders():
    dataset = CUB_Dataset()
    
    all_attributes = dataset.get_attributes()    
    create_folder_from_attribute(all_attributes[0],dataset.get_images_with_attribute)
    
    folder_location = "dataset/images/{}".format(all_attributes[0])
    assert os.path.isdir(folder_location)
    assert len(os.listdir(folder_location))>0
    
    create_random_folder_without_attribute(all_attributes[0],1,dataset.get_images_without_attribute)
    folder_location = "dataset/images/random500_0"
    assert os.path.isdir(folder_location)
    assert len(os.listdir(folder_location))>0
    
if __name__ == "__main__":
    test_cub()
    test_attribute_folders()
