"""
main.py
~~~~~~~

Entry point of the project. Loads the real dataset through the data
pipeline, builds the CNN model, and kicks off training.
"""

from data_loader.loader import Load
from model.cnn import CNN
from model.train import Train


def main():
    # 1. load real train / validation / test data through the pipeline
    data = Load()
    train_data, validation_data, test_data = data.data_pipeline()

    # 2. build the CNN model (untrained, random weights)
    model = CNN()

    # 3. set up the trainer (device is detected automatically inside Train)
    trainer = Train(
        train_data=train_data,
        validation_data=validation_data,
        model=model
    )

    # 4. train for a chosen number of epochs
    trained_model = trainer.fit(epochs=5)

    print("Training finished.")


if __name__ == '__main__':
    main()