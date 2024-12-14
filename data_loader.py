import os
from translate import translate


class Animals_loader:
    """
    Data Loader for the Animal-10 dataset. Contains useful functions that the class can perform on the dataset:
    Loading the dataset/prepping it, datasplitting.
    """

    dataset = []
    x_train, y_train, x_sub_train, y_sub_train, x_val, y_val, x_test, y_test = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    labels = []

    def __init__(self):

        self.dir = "Animals-10/raw-img"

    # Split the data into training, and test set
    def split(self):
        pass

    # Split the data into training: {x_sub_train,y_sub_train, x_validation,y_validation} and test set
    def split_validation(self):
        pass

    # rename to extract or label
    def create_dataset(self):

        print("Creating a label array of the raw-img directory...\n")
        for animal in os.listdir(self.dir):
            self.labels.append([translate[animal], animal])
            class_dir = os.path.join(self.dir, animal)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.dataset.append((file_path, animal))
        print(self.dataset[0])
        return self.dataset
