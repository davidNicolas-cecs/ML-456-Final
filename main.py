import numpy as np
from data_loader import Animals_loader
from PIL import Image
from utils import encode, hot_encode

# Load the animal class
data = Animals_loader()
# create the datsaet (image,label)
dataset = data.create_dataset()

# convert image to array with label=y
img_array, y = data.image_to_array(dataset)

print("type of imaage array", type(img_array))
print("type of individual img", type(img_array[0]))


# converting  to numpyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy


# split the data into training and test set
X_train, X_test, y_train, y_test = data.split(img_array, y)

print("train 1:\n", X_train[0], y_train[0])
# convert labels to numerica value 0-9
y_train_numeric = encode(y_train)
y_train_hot_encoded = hot_encode(y_train_numeric)
print("\nLabels: ", y_train_numeric[0])
# normalize inputs
print(len(X_train))
print("normalized", X_train[1].shape)
# Test
print(y_train_hot_encoded[0])
print("type of x_train set", type(X_train))
print("type of y_train set", type(y_train_hot_encoded))
print("type of individual y", type(y_train_hot_encoded[0]))
sub_train, x_valid, y_sub_train, y_valid = data.val_split(X_train,y_train)



image = Image.fromarray((X_train[0] * 255).astype(np.uint8))


# Save or display the image
image.show()
image.close()

# model 2 will use validation to early stop
# sub_train = x_train[0:50_000]#.reshape(-1, 28 * 28)
# sub_y_train = y_train[0:50_000]

# valid = x_train[:10_000]
# y_valid = y_train[:10_000]
