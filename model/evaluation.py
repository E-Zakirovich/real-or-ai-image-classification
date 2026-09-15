"""
evaluation.py
~~~~~~~~~~~~~

I am going to use this file to caluclate correcness of 
the model. I will use metrics of image classification.
This file will be connected to train.py file.
"""


import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Evaluation:
    def __init__(
            self,
            model, # convolutional neural network is here
            dataloader, # help me to load the eval dataset
            criterion, # cost function
            device, # device (for now, my Macintosh)

    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    """
    helps me to run evaluation
    """

    def run(self):
        self.model.eval()

        # evaluation metrics
        total_loss = 0.0
        total = 0
        all_predictions = []
        all_labels = []

        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            # I need to get outputs from model
            outputs = self.model(images)

            # loss function
            loss = self.criterion(outputs, labels)

            # total loss, total number of operations
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # helper to calculate metrics
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, fscore, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='macro',
            zero_division=0
        )

        return avg_loss, accuracy, precision, recall, fscore, support