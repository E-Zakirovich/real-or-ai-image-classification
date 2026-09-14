"""
methods.py
~~~~~~~~~~~

I created this file to write important methods that used in
loader.py file. Basically, I could write all methods inside
of loader.py, but I want my code must be easy to understand.
"""
from distutils.command.config import config

# load packages
from torchvision import datasets
from torch.utils.data import Subset, DataLoader
from augmentation import Augmentation
import configs


class Methods:

    # image loader
    @staticmethod
    def load_images(
            path, # the location of images
            transform, # transformation type (for train, validation or test)
    ):

        # import images from path
        images = datasets.ImageFolder(
            root = path, # the location of the images
            transform = transform # transformation type (for train, validation or test)
        )

        # return the result
        return images

    # subset maker
    @staticmethod
    def make_subset(
            dataset, # dataset to make a subset
            indices # index of images
    ):
        subset = Subset(
            dataset = dataset, # THE dataset
            indices = indices # indices
        )

        # return the result
        return subset

    @staticmethod
    def load_dataset(
            subset, # we will get the data with subset
            shuffle # true or false for shuffle, better true for train data, otherwise false is better
    ):
        dataset = DataLoader(
            dataset = subset, # subset
            shuffle = shuffle, # true / false
            batch_size = configs.batch_size, # the size of batch (here it is 16)
            num_workers = configs.num_workers, # connection with hardware
        )

        # return the result
        return dataset