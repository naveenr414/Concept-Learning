import numpy as np
import tcav.tcav_examples.image_models.imagenet.imagenet_and_broden_fetcher as fetcher
import os
import urllib
import socket
from subprocess import DEVNULL, STDOUT, check_call
import tensorflow as tf

def unzip_tar(file_location,folder_location):
    """Unzip tar file at file_location into folder at folder_location
    
    Arguments:
        file_location: Location of the class_name.tar file; filled with images
        folder_location: Folder into which file_location should be extracted into
    
    Returns:
        None
    
    Side Effects:
        Extracts file_location into folder_location
    """
    
    check_call(["mkdir -p {}".format(folder_location)],stdout=DEVNULL, stderr=STDOUT,shell=True)
    check_call(["rm -f {}/*".format(folder_location)],stdout=DEVNULL, stderr=STDOUT,shell=True)
    check_call(["tar xvf {} --directory {}".format(file_location,folder_location)],
              stdout=DEVNULL,
              stderr=STDOUT, 
              shell=True)
    check_call(["rm {}".format(file_location)],stdout=DEVNULL,stderr=STDOUT, shell=True)

def keep_n_files(folder_location,num_files):
    """Keep only num_files at folder_location, deleting all others
    
    Arguments:
        folder_location: Location of the folder from which to delete data
        num_files: Number of files to delete; all others will be deleted
    
    Returns:
        None
        
    Side Effects:
        Deletes all but num_files at folder_location
    """

    check_call(["ls  {}/* | head -n -{} | xargs -d '\n' rm -f".format(folder_location,num_files)], 
               shell=True)

def fetch_imagenet_class(path, class_name, images_per_class, imagenet_dataframe):
  """Fetch all images from an imagenet class
  
  Arguments:
      path: Location of the dataset folder; all data will be stored into a new folder there
      class_name: ImageNet class_name for images; images are stored at path/class_name
      images_per_class: How many image files should be stored per class
      imagenet_dataframe: Dataframe mapping ImageNet classes to dataframes

  Returns:
    None
    
  Side Effects:
    Creates new folder with ImageNet images by downloading from URL
  """
    
  if imagenet_dataframe is None:
    raise tf.errors.NotFoundError(
        None, None,
        "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
    )
  # To speed up imagenet download, we timeout image downloads at 5 seconds.
  socket.setdefaulttimeout(5)
    
  # Check to see if this class_Name exists
  dataframe_results = imagenet_dataframe[imagenet_dataframe["class_name"] == class_name]
  if len(dataframe_results) == 0:
    tf.compat.v1.logging.info("Unable to find {} in dataframe; failed to download data".format(class_name))
    return
  class_id = dataframe_results["synid"].values[0]

    
  url_for_image = "https://image-net.org/data/winter21_whole/{}.tar".format(class_id)
  file_location = path +"/"+class_name.lower().replace(" ","_")+".tar"
  folder_location = path + "/"+class_name.lower().replace(" ","_")
    
  tf.compat.v1.logging.info("Fetching imagenet data for " + class_name)
  tf.io.gfile.makedirs(folder_location)
  tf.compat.v1.logging.info("Saving images at " + folder_location)
  
  try:
    # Download from url_for_image 
    urllib.request.urlretrieve(url_for_image, file_location)
    unzip_tar(file_location,folder_location)
    keep_n_files(folder_location,images_per_class)
    
  except Exception as e:
    tf.compat.v1.logging.error("Problem downloading imagenet class. Exception was " +
                             str(e) + " for URL "+url_for_image)

  number_of_images = len(os.listdir(folder_location))
  tf.compat.v1.logging.info("Downloaded {} images for {}".format(number_of_images,class_name))
    
def download_imagenet(imagenet_classes,images_per_class):
    """Download images from a list of ImageNet Classes
    
    Arguments:
        imagenet_classes: List of imagenet classes
        images_per_class: How many images should be kept for each ImageNet class
    
    Return:
        None
    
    Side Effects:
        Creates new folders in dataset for each ImageNet class
    """
    
    tf.compat.v1.logging.info("Downloading imagenet_classes {}".format(imagenet_classes))
    
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./dataset/meta/imagenet_url_map.csv")
    source_dir = "./dataset/images"

    for image in imagenet_classes:
        tf.compat.v1.logging.info("Downloading class {}".format(image))
        fetch_imagenet_class(source_dir, image, images_per_class, imagenet_dataframe)
