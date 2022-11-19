from src.download_data import *
import os

def test_imagenet_load():
    """Tests the download_imagenet function"""
    
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./dataset/meta/imagenet_url_map.csv")
    assert len(imagenet_dataframe[imagenet_dataframe["class_name"] == "zebra"]) == 1
    
    assert download_imagenet(["zebra","goldfish"],1) == None
    assert len(os.listdir("./dataset/zebra")) == 1
    assert len(os.listdir("./dataset/goldfish")) == 1

    assert download_imagenet(["zebra","goldfish"],5) == None
    assert len(os.listdir("./dataset/zebra")) == 5
    assert len(os.listdir("./dataset/goldfish")) == 5