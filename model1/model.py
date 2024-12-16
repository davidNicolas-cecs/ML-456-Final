from data_loader import Animals_loader


def prep():
    # Prepping the data
    animal_loader = Animals_loader()
    img_dataset = animal_loader.create_dataset()
    x, y = animal_loader.image_to_array(img_dataset)
    return animal_loader.split(x, y)


def build_model():
    pass


def main():
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prep()


if __name__ == "__main__":
    print(
        "Starting Model 1 dataset\n: * Contains 10 layers, 8 Conv2D, and 2 Dense Layers"
    )
    main()
