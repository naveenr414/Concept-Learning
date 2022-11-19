from src.download_data import *
import os
import tcav.tcav_examples.image_models.imagenet.imagenet_and_broden_fetcher as fetcher

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
tf.autograph.set_verbosity(0)


def test_load_imagenet():
    """Tests the download_imagenet function"""
    
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./dataset/meta/imagenet_url_map.csv")
    assert len(imagenet_dataframe[imagenet_dataframe["class_name"] == "zebra"]) == 1
    
    assert download_imagenet(["zebra","goldfish"],1) == None
    assert len(os.listdir("./dataset/zebra")) == 1
    assert len(os.listdir("./dataset/goldfish")) == 1

    assert download_imagenet(["zebra","goldfish"],5) == None
    assert len(os.listdir("./dataset/zebra")) == 5
    assert len(os.listdir("./dataset/goldfish")) == 5
    
if __name__ == "__main__":
    test_load_imagenet()