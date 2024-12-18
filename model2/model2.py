import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from ..data_loader import Animals_loader
from ..utils import encode, hot_encode

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import RMSprop, l1

def prep():
  # Prepping the data
  animal_loader = Animals_loader()
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
      padding="same",
      strides=(2,2),
      input_shape=(32,32,3)
    )
  )
  # every layer has convultion -> batch norm -> maxpooling (no dropout)
  for i in range(hp.Int("num_layers_conv", 3, 10)):
    model.add(
      Conv2D(
        filters=hp.Int(f"filters_{i-1}", min_value=32, max_value=512, step=32),
        kernel_size=(3,3),
        activation="relu",
        padding="same",
        strides=(1,1)
      )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
  # Flatten for Dense Layers
  model.add(Flatten())
  for j in range(5):
    model.add(
      Dense(
        units=hp.Int(f'dense_units{j}', min_value=32, max_value=512, step=32),
        activation='sigmoid',
        kernal_regularization=l1(.1)
      )
    )
  # output layer
  model.add(Dense(10,activation='softmax'))  
  
  model.compile(optimizer=keras.optimizers.Adam(
     learning_rate=1e-3,
     beta_1=.9,
     beta_2=.999,
  ),
    loss = 'categorical_crossentropy', metrics=['accuracy'])
  return model

def main():
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prep()


if __name__ == "__main__":
    print(
        "Starting Model 2 dataset\n: * Contains ?? layers, ?? Conv2D, and ??? Dense Layers"
    )
    main()