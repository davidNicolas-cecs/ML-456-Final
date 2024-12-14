# ML-456-Final

### How to Run

Prerequisites:

- ensure you have the latest version of python3
- ensure you have the latest version of pip

To run our models all you need to do is activate the virtual enviorment called `ml-env`. This contains all the libraries and dependencies the project needs in order to run.

We also provide the requirements.txt file of our dependencies.

Run each individual model to see its accuracy as well as a sample of the first few images guessed and its true value

### Components of the project

##### data_loader.py:

This python file contains the class 'Animal_loader' which houses small useful functions that both models will need to train and test their data. It reduces redundancy in our models code, allowing us to easily call this class to:

- initialize labels
- split data into training and test set

##### model1.py

### File Structure

- Animals-10 contains the raw images of the Animals-10 dataset
- model1 contains the python file for model1
- model2 contains the python file for model2
- ml-env is the virtual environment to run the project
