from src.dataset import *

def test_cub():
    all_attributes = get_cub_attributes()
    assert len(all_attributes) == 112
    
    assert len(load_cub_split("train")) > 0
    assert len(load_cub_split("val")) > 0
    assert len(load_cub_split("test")) > 0
    
    train = load_cub_split("train")
    assert 'img_path' in train[0]
    assert 'attributes' in train[0]

if __name__ == "__main__":
    test_cub()