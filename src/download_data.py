import numpy as np
import tcav.tcav_examples.image_models.imagenet.imagenet_and_broden_fetcher as fetcher
import os
import urllib
import socket
from subprocess import DEVNULL, STDOUT, check_call
import tensorflow as tf

def fetch_imagenet_class(path, class_name, images_per_class, imagenet_dataframe):
  if imagenet_dataframe is None:
    raise tf.errors.NotFoundError(
        None, None,
        "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
    )
  # To speed up imagenet download, we timeout image downloads at 5 seconds.
  socket.setdefaulttimeout(5)

  tf.compat.v1.logging.info("Fetching imagenet data for " + class_name)
  concept_path = os.path.join(path, class_name)
  tf.io.gfile.makedirs(concept_path)
  tf.compat.v1.logging.info("Saving images at " + concept_path)

  # Check to see if this class name exists.
  dataframe_results = imagenet_dataframe[imagenet_dataframe["class_name"] == class_name]
  if len(dataframe_results) == 0:
    tf.compat.v1.logging.info("Unable to find {} in dataframe; failed to download data".format(class_name))
    return
  class_id = dataframe_results["synid"].values[0]
  
  url_for_image = "https://image-net.org/data/winter21_whole/{}.tar".format(class_id)
  file_location = path +"/"+class_name+".tar"
  folder_location = path + "/"+class_name

  try:
    # Download from url_for_image 
    saving_path = os.path.join(path, class_name+".tar")
    urllib.request.urlretrieve(url_for_image, file_location)
    
    # Download files and move to directory  
    check_call(["mkdir -p {}".format(folder_location)],stdout=DEVNULL, stderr=STDOUT,shell=True)
    check_call(["rm {}/*".format(folder_location)],stdout=DEVNULL, stderr=STDOUT,shell=True)
    check_call(["tar xvf {} --directory {}".format(file_location,folder_location)],
              stdout=DEVNULL,
              stderr=STDOUT, 
              shell=True)
    check_call(["ls  {}/* | head -n -{} | xargs -d '\n' rm -f".format(folder_location,images_per_class)], 
              shell=True)
    check_call(["rm {}".format(file_location)],stdout=DEVNULL,stderr=STDOUT, shell=True)
    
  except Exception as e:
    tf.compat.v1.logging.info("Problem downloading imagenet class. Exception was " +
                             str(e) + " for URL "+url_for_image)

  number_of_images = len(os.listdir(folder_location))
  tf.compat.v1.logging.info("Downloaded {} images for {}".format(number_of_images,class_name))
    
def download_imagenet(imagenet_classes,images_per_class):
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./dataset/meta/imagenet_url_map.csv")
    
    source_dir = "./dataset"

    for image in imagenet_classes:
        fetch_imagenet_class(source_dir, image, images_per_class, imagenet_dataframe)