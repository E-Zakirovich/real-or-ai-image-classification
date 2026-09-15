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
            train_data, # with this data my convolutional neural network will learn
            validation_data, # with this data, I am going to learn the correctness of the network
            model # actual convolutional neural network model
            ):

        self.train_data = train_data # special variable for train data
        self.validation_data = validation_data # special variable for train data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) # speical variable for model itself        

        # loss and optmization frunctions are imported
        self.loss_function = neural_networks.CrossEntropyLoss()
        self.optimizer = optimize.Adam(
            self.model.parameters(), 
            lr = configs.learning_rate
        )

    def fit(self, epochs : int):

        for epoch in range(epochs):

            # starting part of the training
            self.model.train()

            # running loss will demonstrate loss of each epoch, correct and total variables will help me to calculate running loss
            running_loss = 0.0
            correct = 0.0
            total = 0.0

            for image, label in self.train_data:

                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(image)

                loss = self.loss_function(outputs, label)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item() * label.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            print(
                f"""
                    train loss: {train_loss}
                    train accuracy: {train_accuracy}
                """
            )

        return self.model # 67 67 67 67 (six seven meme. i dunno why but this meme came to my mind right now with no reason).