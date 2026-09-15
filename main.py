import torch
from model.cnn import CNN

def main():
    model = CNN()

    # dummy batch: batch_size=4, channels=3, height=256, width=256
    dummy_input = torch.randn(4, 3, 256, 256)

    output = model(dummy_input)

    print(output.shape)  # should be [4, 2] if output layer has 2 neurons
    print(output)

if __name__ == '__main__':
    main()