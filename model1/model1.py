
from data_loader import Animals_loader
from utils import encode, hot_encode

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

def build_model(hp):
  model = Sequential()
  # First layer Input size
  model.add(
      Conv2D(
        filters=hp.Int(f'filters_{0}', min_value=32, max_value=512, step=32),
        kernel_size=(3, 3),
        activation="relu",
        padding="valid",
        strides=(1,1),
        input_shape=(128,128,3)
    )
  )
  for i in range(hp.Int("num_layers_conv", 3, 10)):
      model.add(
          Conv2D(
              filters=hp.Int(f"filters_{i-1}", min_value=32, max_value=512, step=32), #32
              kernel_size=(3,3), 
              activation="relu",
              padding="valid",
              strides=(1,1)
          )
      )
      # every 2 Conv2d layers, add a batch norm and maxpooling and possible dropout
      if i % 2 == 0:
          model.add(BatchNormalization())
          model.add(MaxPooling2D(2,2))
          # Optional Dropout
          if hp.Boolean(f"dropout_after_pool_{i}", default=True):
              model.add(Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.2, max_value=0.5, step=0.1)))
      
  # Flatten for Dense Layers
  model.add(Flatten())
      
  for j in range(5):
      model.add(
          Dense(
                units=hp.Int(f'dense_units{j}', min_value=32, max_value=512, step=32),
                activation='sigmoid',
                kernel_regularizer=l1(.1)
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






def main(animal_loader):
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prep(animal_loader)
    max_imgs = len(x_train)
    print(x_train[0], y_train[0])
    # Split into validation set
    
    sub_train, x_valid, y_sub_train, y_valid = animal_loader
    
    # Building the model to run trials to test best model    
    build_model(kt.HyperParameters())
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',# maybe no early stop
        max_trials=2, # 15
    )
    '''
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights = False,
        mode = "min"
    )
    '''
    
    # search for best model
    tuner.search(x_train, y_train, epochs=4, validation_data = (y_train,y_test), batch_size=64) # change to actual val later


if __name__ == "__main__":
    animal_loader = Animals_loader()

    print(
        "\nStarting Model 1\n: * Contains ?? layers\n, *Every 2 layers of Conv2D, a batch norm and maxpooling layer is applied\n,\n *Possible dropout layer * ??? Dense Layers\m *No variation in padding\n*No variation in strides, use l1 regularization"
    )
    main(animal_loader)