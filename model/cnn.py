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

        # relu
        self.relu = neural_networks.ReLU()

        # pooling
        self.pool = neural_networks.MaxPool2d(
            kernel_size = configs.kernel_size_for_pooling,
            stride = configs.stride_size_for_pooling
        )

        # connection between input and hidden layer
        self.fully_connected_one = neural_networks.Linear(
            configs.input_layer,
            configs.hidden_layer
        )

        # dropout
        self.drop = neural_networks.Dropout(
            configs.dropout
        )

        # connection between hidden layer and output layer
        self.fully_connected_two = neural_networks.Linear(
            configs.hidden_layer,
            configs.output_layer
        )

    # feed forward 
    def forward(self, x):

        # first layer
        x = self.pool(self.relu(self.first_batch_normalization(self.first_convolutional_layer(x))))

        # second layer
        x = self.pool(self.relu(self.second_batch_normalization(self.second_convolutional_layer(x))))

        # third layer
        x = self.pool(self.relu(self.third_batch_normalization(self.third_convolutional_layer(x))))

        # fourth layer
        x = self.pool(self.relu(self.fourth_batch_normalization(self.fourth_convolutional_layer(x))))

        # fifth layer
        x = self.pool(self.relu(self.fifth_batch_normalization(self.fifth_convolutional_layer(x))))

        # neural networks part 
        x = x.view(x.size(0), -1)

        # nnet building part
        # connection between input and hidden layer
        x = self.relu(self.fully_connected_one(x))

        # dropout
        x = self.drop(x)

        # connection between hidden and output layer
        x = self.fully_connected_one(x)

        # return the result
        return x