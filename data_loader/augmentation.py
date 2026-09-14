"""
augmentation.py
~~~~~~~~~~~~~~~~

Augmentation part of data loading is - making random rotations, flipping,
changing  the size of images  into same  size, normalize  the images and
making tensor from loaded images.
"""

# load packages
from torchvision import transforms
import configs


class Augmentation:

    @staticmethod
    def transform_for_train_dataset():

        # transformation code for train data
        result = transforms.Compose([

            # change the size of the image
            transforms.Resize(
                (
                    configs.image_size, # amount
                    configs.image_size, # amount
                )
            ),

            # flip the image horizontally in order to avoid overfitting
            transforms.RandomHorizontalFlip(
                p = configs.horizontal_flip, # amount
            ),

            # making tensor from processed images
            transforms.ToTensor(),

            # normalization part of the data
            transforms.Normalize(
                mean = configs.mean, # mean
                std = configs.std, # std
            )

        ])

        # returning the result
        return result

    @staticmethod
    def transform_for_validation_and_test_dataset():

        # transformation code for validation and test data
        result = transforms.Compose([

            # change the size of the image
            transforms.Resize(

                (
                    configs.image_size, # amount
                    configs.image_size # amount
                )
            ),

            # make a tensor from imported image
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean = configs.mean, # mean
                std = configs.std # std
            )
        ])

        # returning the result
        return result