# ML-456-Final

`The following was written to run on vscode with python.`

## How to Run

### Prerequisites:

- ensure you have the latest version of python or python3
- ensure you have the latest version of pip

Check:

```
pip --version
python --version
python3 --version
```

## Activate the Environment

The virutal environment contains all the libraries and dependencies the project needs in order to run. Follow the directions depending on the operating system you are running.

### For Linux

To run our models on linux all you need to do is activate the virtual enviorment called `ml-env`.

```bash
source ml-env/bin/activate
```

### For Windows

You may need to allow for a virtual environment to be activated on windows by running:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

You can then activate the virtual environment called `windows-ml-env`:

```bash
windows-ml-env/Script/activate
```

We also provide the requirements.txt file of our dependencies in case you have some other method to install the dependencies.

`Regardless of operating system if the virtual enviorment is missing some imports run:`

```
pip install --upgrade -r requirements.txt
```

# Note:

For the first half of development we used CPU, and later installed and ran with GPU, ensure you have working GPU, as we did not do any catches for gpu not present, tensor may default to CPU if no gpu detected.

## To Run a Model

Run each individual model to see its accuracy as well as a sample of the first few images guessed and its true value.

To run a model:

Command to run model 1

```bash
python3 -m model.model1
or
python -m model.model1
```

## Components of the Project

#### data_loader.py:

This python file contains the class 'Animal_loader' which houses small useful functions that both models will need setup the data for training and testing. It reduces redundancy in our models code, allowing us to easily call this class to

- initialize labels
- split data into training and test set
- normalize each image

#### utils.py

This python file contains the translate.py code given in the dataset, as well as functions to

- tranform label output to numerical
- hot encode the outputs for testing

#### model1.py

This will be a brief overview of the model, read the report to learn more.
`model1.py` uses keras tuner to pick from choices we provide the model with.

#### model_simple

This will be a brief overview of the model, read the report to learn more.
`model_simple.py` is as the name states a simpler model of model1 without a tuner, less layers, and the architecture is more strict.

## File Structure

- Animals-10 contains the raw images of the Animals-10 dataset
- model contains the python file for model1
- ml-env `or` windows-ml-env is the virtual environment to run the project