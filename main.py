from data_loader import Animals_loader
from PIL import Image


data = Animals_loader()

dataset = data.create_dataset()


img_array, y = data.image_to_array(dataset)

print(len(y))
print(len(img_array))


X_train, X_test, y_train, y_test = data.split(img_array, y)

print(len(X_train), len(X_test))

print(X_train[0], y_train[0])

image = Image.fromarray(X_train[0])

# Save or display the image
image.show()
