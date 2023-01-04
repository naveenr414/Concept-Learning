from src.dataset import *

def test_cub():
    all_attributes = get_cub_attributes()
    assert len(all_attributes) == 112
    
    assert len(load_cub_split("train")) > 0
    assert len(load_cub_split("val")) > 0
    assert len(load_cub_split("test")) > 0
    
    train = load_cub_split("train")
    assert 'img_path' in train[0]
    assert 'attribute_label' in train[0]

    with_attribute = get_cub_images_by_attribute(all_attributes[0])
    without_attribute = get_cub_images_without_attribute(all_attributes[0])
    
    assert len(with_attribute)>0
    assert len(without_attribute)>0
    assert with_attribute[0] not in without_attribute
    assert without_attribute[0] not in with_attribute
    
def test_attribute_folders():
    all_attributes = get_cub_attributes()
    create_folder_from_attribute(all_attributes[0])
    
    folder_location = "dataset/images/{}".format(all_attributes[0])
    assert os.path.isdir(folder_location)
    assert len(os.listdir(folder_location))>0
    
    create_random_folder_without_attribute(all_attributes[0],1)
    folder_location = "dataset/images/random500_0"
    assert os.path.isdir(folder_location)
    assert len(os.listdir(folder_location))>0
    
if __name__ == "__main__":
    test_cub()
    #test_attribute_folders()
