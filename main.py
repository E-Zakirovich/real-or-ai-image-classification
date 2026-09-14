from data_loader.loader import Load

def main():
    data = Load()
    train_data, validation_data, test_data = data.data_pipeline()

    images = []
    labels = []
    for img_batch, label_batch in train_data:
        for i in range(img_batch.size(0)):
            images.append(img_batch[i])
            labels.append(label_batch[i])
            if len(images) == 10:
                break
        if len(images) == 10:
            break

    print(images[0].shape)
    print(labels[:10])

if __name__ == '__main__':
    main()