"""
loader.py
~~~~~~~~~

With  this file I will return train, validation and  test dataset
to  artificial  model. Random  seed, random split  operations are
also written here. Btw, I will use augmentation.py and methods.py
to make worked datapipeline.
"""

# load packages
import torch
from torch.utils.data import random_split
from augmentation import Augmentation
from methods import Methods
import configs

# I am importing Augmentation
augmentation = Augmentation()

# I am importing methods
methods = Methods()


class Load:

    @staticmethod
    def __load_data():

        # load train images
        train_images = methods.load_images(
            path = configs.train_and_validation_data_path, # location for train dataset
            transform = augmentation.transform_for_train_dataset() # transformation method for import images
        )

        # load validation images
        validation_images = methods.load_images(
            path=configs.train_and_validation_data_path,  # location for validation dataset
            transform=augmentation.transform_for_validation_and_test_dataset() # transformation method for import images
        )

        # load test images
        test_images = methods.load_images(
            path=configs.test_data_path,  # location for test dataset
            transform=augmentation.transform_for_validation_and_test_dataset() # transformation method for import images
        )


    @staticmethod
    def data_pipeline():

        train_data, validation_data, test_data = Load.__load_data()

        return train_data, validation_data, test_data