class Animals_loader:
    """
    Data Loader for the Animal-10 dataset. Contains useful functions that the class can perform on the dataset:
    Loading the dataset/prepping it, datasplitting.
    """

    x_train, y_train, x_sub_train, y_subtrain, x_val, y_val, x_test, y_test = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    def __init__(self):

        pass

    # Split the data into training, and test set
    def split(self):
        pass

    # Split the data into training: {x_sub_train,y_sub_train, x_validation,y_validation} and test set
    def split_validation(self):
        pass
