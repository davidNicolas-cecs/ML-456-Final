# add early stopping, extra strides options
from ..data_loader import Animals_loader
from ..utils import encode, hot_encode
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import RMSprop

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
        padding=hp.Choice("padding", ['same', 'valid']),
        strides=(1,1),
        input_shape=(32,32,3)
    )
  )
  for i in range(hp.Int("num_layers_conv", 3, 7)):
      pass

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