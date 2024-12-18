
import numpy as np
from data_loader import Animals_loader
from utils import encode, hot_encode
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.regularizers import l1
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

def prep(animal_loader):
    # Prepping the data using Animal Loader class and utilty functions 
    
    img_dataset = animal_loader.create_dataset()
    x, y = animal_loader.image_to_array(img_dataset)
    y = encode(y)
    y = hot_encode(y)
    print(y)
    return animal_loader.split(x, y)

def build_model():
  model = Sequential()
  # First layer Input size
  model.add(
      Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="valid",
        strides=(1,1),
        input_shape=(128,128,3)
    )
  ) 
  model.add(
          Conv2D(
              filters=128, #32
              kernel_size=(3,3), 
              activation="relu",
              padding="valid",
              strides=(1,1)
          )
      )
  model.add(BatchNormalization())
  model.add(MaxPooling2D(2,2))        
  model.add(
          Conv2D(
              filters=64, #32
              kernel_size=(3,3), 
              activation="relu",
              padding="valid",
              strides=(1,1)
          )
      )
  model.add(BatchNormalization())
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(rate=.2))
  model.add(
          Conv2D(
              filters=128, #32
              kernel_size=(3,3), 
              activation="relu",
              padding="valid",
              strides=(1,1)
          )
      )
  
      
   
  # Flatten for Dense Layers
  model.add(Flatten())
      
  for j in range(5):
      model.add(
          Dense(
                units=(64 + j * 32),
                activation='sigmoid',
                kernel_regularizer=l1(.01)
            )
        )

  # output layer
  model.add(Dense(10,activation='softmax'))  
  
  
  # compile model
  model.compile(optimizer=keras.optimizers.Adam(
      learning_rate=1e-3,
      beta_1=.9,
      beta_2=.999,
  ),
    loss = 'categorical_crossentropy', metrics=['accuracy'])
  return model


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()



def main():
    animal_loader = Animals_loader()
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prep(animal_loader)
    # Split into validation set
    sub_train, x_valid, y_sub_train, y_valid = animal_loader.val_split(x_train,y_train)
    
    # Building the model to run trials to test best model    
    
    '''
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights = False,
        mode = "min"
    )
    '''
        
    model = build_model()
    history = model.fit(
        sub_train, y_sub_train, 
        batch_size=256, epochs=1, 
        validation_data=(x_valid, y_valid),
    )

    # Print model summary
    model.summary()
    
    # search for best model
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plot_training_history(history)
    #model.save('simple_model1.h5')

    
if __name__ == "__main__":
    print(
        "\nStarting Model 1:\n - Contains ?? layers\n - Every 2 layers of Conv2D, a batch norm and maxpooling layer is applied\n - Possible dropout layer\n - ??? Dense Layers \n - No variation in padding\n - No variation in strides, use l1 regularization\n"
        +"-----------------------------------------------"
    )
    main()