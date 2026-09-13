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
    def __init__(self, train, validation, test):

        self.train_path = train # path to train dataset
        self.validation_path = validation # path to validation dataset
        self.test_path = test # path to test dataset

    @staticmethod
    def transform_for_train_dataset(path):

        # transformation code for transforms
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

            # normalization part of the data
            transforms.Normalize(
                mean = configs.mean, # mean
                std = configs.std, # std
            ),

            # making tensor from processed images
            transforms.ToTensor(),
        ])

        # returning the result
        return result

    @staticmethod
    def transform_for_validation_and_test_dataset(path):
        return ...

    def augmented_data(self):

        train_dataset = transform_for_train_dataset(self.train_path)
        validation_dataset = transform_for_validation_and_test_dataset(self.validation_path)
        test_dataset = transform_for_validation_and_test_dataset(self.test_path)

        return train_dataset, validation_dataset, test_dataset