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
from .augmentation import Augmentation
from .methods import Methods
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

        # generate seed
        just_seed = torch.Generator().manual_seed(
            configs.seed
        )

        # splitting them according to their indices
        train_indices, validation_indices = random_split(
            train_images, # images
            lengths = [
                configs.train_split, # 0.9
                configs.validation_split, # 0.1
            ],
            generator = just_seed # seed = 42
        )

        # make a subset for train data
        train_subset = methods.make_subset(
            train_images,  # src for subset
            indices=train_indices,  # indices
        )

        # make a subset for validation data
        validation_subset = methods.make_subset(
            validation_images,  # src for subset
            indices=validation_indices,  # indices
        )

        # load the train dataset
        train_data = methods.load_dataset(
            train_subset,
            batch_size=configs.batch_size,
            num_workers=configs.num_workers,
            shuffle=True
        )

        # load the validation dataset
        validation_data = methods.load_dataset(
            validation_subset,
            batch_size=configs.batch_size,
            num_workers=configs.num_workers,
            shuffle=False
        )

        # load the test dataset
        test_data = methods.load_dataset(
            test_images,
            batch_size=configs.batch_size,
            num_workers=configs.num_workers,
            shuffle=False
        )

        return train_data, validation_data, test_data

    @staticmethod
    def data_pipeline():

        train_data, validation_data, test_data = Load.__load_data()

        return train_data, validation_data, test_data