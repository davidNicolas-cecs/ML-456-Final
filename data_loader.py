import os

import numpy as np
from sklearn.model_selection import train_test_split
from utils import translate
from PIL import Image


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
        
    def val_split(self, x,y):
        return train_test_split(x,y,test_size=.2,random_state=1)

    # Split the data into training, and test set
    def split(self, X, Y):
        """split _summary_

        Parameters
        ----------
        X : Array
            images as arrays
        Y : Array
            corresponding labels to images

        Returns
        -------
        train & test data
            split data into test and train set
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, shuffle=True
        )
        return X_train, X_test, y_train, y_test

    # Split the data into training: {x_sub_train,y_sub_train, x_validation,y_validation} and test set
    def split_validation(self, data):
        pass

    # rename to extract or label
    def create_dataset(self):
        """create_dataset: binds image with corresponding label

        Returns
        -------
        Array
            [[image,label],]
        """
        print("Creating a label array of the raw-img directory...\n")
        for animal in os.listdir(self.dir):
            self.labels.append([translate[animal], animal])
            class_dir = os.path.join(self.dir, animal)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.dataset.append((file_path, animal))
        return self.dataset

    def image_to_array(self, data):
        """image_to_array: converts image to numpy array

        Parameters
        ----------
        data : Array
            Array of images with label

        Returns
        -------
        Array
            image array and corresponding label
        """
        img_size = (128, 128)
        img_array = []
        label_array = []
        print("Converting a label image array to numpy arrays...\n")
        # image_size = (128, 128)
        for image in data:
            with Image.open(image[0]) as img:
                img = img.convert("RGB")
                # normalizing
                img = img.resize(img_size)
                img_arr = np.array(img)
                img_arr = img_arr / 255.0

            label = image[1]
            img_array.append(img_arr)
            label_array.append(label)
        return np.array(img_array), label_array
