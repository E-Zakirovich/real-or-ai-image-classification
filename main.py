from data_loader.loader import Load

data = Load()

train_data, validation_data, test_data = data.data_pipeline()

first_10 = [train_data[i] for i in range(10)]

print(first_10)