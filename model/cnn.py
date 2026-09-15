"""
cnn.py
~~~~~~~

This file is core of the project. To train my model, basically I used 
convolutional neural network. My network includes five layers, one extra 
paddings to improve performance of the model.
"""

# import libraries
import torch.nn as neural_networks
import configs

class CNN(neural_networks.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # first concolutional layer
        self.first_convolutional_layer = neural_networks.Conv2d(
            in_channels = configs.in_out_channels[0],
            out_channels = configs.in_out_channels[1],
            kernel = configs.kernel_size,
            stride = configs.stride_size,
            padding = configs.padding_size
        )

        # first batch normalization
        self.first_batch_normalization = neural_networks.BatchNorm2d(configs.in_out_channels[1])

        # second concolutional layer
        self.second_convolutional_layer = neural_networks.Conv2d(
            in_channels = configs.in_out_channels[1],
            out_channels = configs.in_out_channels[2],
            kernel = configs.kernel_size,
            stride = configs.stride_size,
            padding = configs.padding_size
        )

        # second batch normalization
        self.second_batch_normalization = neural_networks.BatchNorm2d(configs.in_out_channels[2])

        # third concolutional layer
        self.third_convolutional_layer = neural_networks.Conv2d(
            in_channels = configs.in_out_channels[2],
            out_channels = configs.in_out_channels[3],
            kernel = configs.kernel_size,
            stride = configs.stride_size,
            padding = configs.padding_size
        )

        # third batch normalization
        self.third_batch_normalization = neural_networks.BatchNorm2d(configs.in_out_channels[3])

        # fourth concolutional layer
        self.fourth_convolutional_layer = neural_networks.Conv2d(
            in_channels = configs.in_out_channels[3],
            out_channels = configs.in_out_channels[4],
            kernel = configs.kernel_size,
            stride = configs.stride_size,
            padding = configs.padding_size
        )

        # fourth batch normalization
        self.fourth_batch_normalization = neural_networks.BatchNorm2d(configs.in_out_channels[4])

        # fifth concolutional layer
        self.fifth_convolutional_layer = neural_networks.Conv2d(
            in_channels = configs.in_out_channels[4],
            out_channels = configs.in_out_channels[5],
            kernel = configs.kernel_size,
            stride = configs.stride_size,
            padding = configs.padding_size
        )

        # fifth batch normalization
        self.fifth_batch_normalization = neural_networks.BatchNorm2d(configs.in_out_channels[5])