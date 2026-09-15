"""
configs.py
~~~~~~~~~~~

I need to store static settings to somewhere, so I created
configs.py file.
"""

# dataset paths
train_and_validation_data_path = "./data/train"
test_data_path = "./data/test"

# augmentation settings
image_size = 256
horizontal_flip = 0.5
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 16
num_workers = 2
seed = 42
train_split = 0.9
validation_split = 0.1

# convolutuional neural network settings
in_out_channels = [8, 16, 32, 64, 128, 256]
kernel_size = 3
stride_size = 1
padding_size = 1
