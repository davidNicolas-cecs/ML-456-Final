# ML-456-Final

## How to Run

### Prerequisites:

- ensure you have the latest version of python3
- ensure you have the latest version of pip

Check:

```
pip --version
python3 --version
```

## Activate the enviorment

To run our models all you need to do is activate the virtual enviorment called `ml-env`. This contains all the libraries and dependencies the project needs in order to run.
To run on macos/linux:

```bash
source ml-env/bin/activate
```

If you are running on windows the following command is used:

```bash
ml-env/Script/activate
```

We also provide the requirements.txt file of our dependencies in case you have some other method to install the dependencies.

If the virtual enviorment is missing some imports run:

```
pip install --upgrade -r requirements.txt
```

# Note:

For the first half of development we used CPU, and later installed and ran with GPU, ensure you have working GPU, as we did not do any catches for gpu not present, tensor may default to CPU if no gpu detected.

## To run a model

Run each individual model to see its accuracy as well as a sample of the first few images guessed and its true value.

To run a model:

Command to run model 1

```bash
python3 -m model1.model1
or
python -m model1.model1
```

## Components of the project

##### data_loader.py:

This python file contains the class 'Animal_loader' which houses small useful functions that both models will need setup the data for training and testing. It reduces redundancy in our models code, allowing us to easily call this class to

- initialize labels
- split data into training and test set
- normalize each image

#### utils.py

This python file contains the translate.py code given in the dataset, as well as functions to

- tranform label output to numerical
- hot encode the outputs for testing

#### model1.py

This will be a brief overview of the model, read the report to learn more :).
Model 1 uses keras tuner to pick from choices we provide the model with.

### File Structure

- Animals-10 contains the raw images of the Animals-10 dataset
- model1 contains the python file for model1
- model2 contains the python file for model2
- ml-env is the virtual environment to run the project
