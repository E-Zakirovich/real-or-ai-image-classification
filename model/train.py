"""
train.py
~~~~~~~~~

I am going to use this file to train my convolutional neural
netoworks. This file is connected to main.py and cnn.py files.
"""
import torch
import numpy
import torch.optim as optimize
import torch.nn  as neural_networks
import configs

class Train:

    def __init__(
            self, 
            device, # the device where I am going to train my mode
            train_data, # with this data my convolutional neural network will learn
            validation_data, # with this data, I am going to learn the correctness of the network
            model # actual convolutional neural network model
            ):

        self.train_data = train_data # special variable for train data
        self.validation_data = validation_data # special variable for train data
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.model = model.to(device) # speical variable for model itself        

        # loss and optmization frunctions are imported
        loss_function = neural_networks.CrossEntropyLoss()
        optimizer = optimize.Adam(

        )

    