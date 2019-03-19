import os
import numpy as np

from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import model_from_json
from keras import models
from keras import backend as K

# Load image
img = load_img("models/thanh.jpg", target_size=(150, 150))
img_arr = img_to_array(img) / 255
img_arr = img_arr.reshape((1,) + img_arr.shape)
# plt.imshow(img_arr[0])
# plt.show()

# Load model
json_file = open("models/face_model.json", "r")
json_string = json_file.read()
json_file.close()
model = model_from_json(json_string)
model.load_weights("models/weights.25.h5")

print(model.summary())

# Get layer
inp = model.input
outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=inp, outputs=outputs)
activations = activation_model.predict(img_arr)


# test
# plt.imshow(img_arr[0])
# plt.show()
# plt.imshow(activations[0][0])
# plt.show()


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    n = activation.shape[-1]
    row_size = min(row_size, n // col_size)
    activation_index = 0
    fig, ax = plt.subplots(row_size,
                           col_size,
                           figsize=(col_size, row_size),
                           sharex=True,
                           sharey=True,
                           gridspec_kw={'wspace': 0.02, 'hspace': 0.02, "left": 0, "right": 1, "bottom": 0, "top": 1, }
                           )
    for row in range(row_size):
        for col in range(col_size):
            if activation_index >= n:
                break

            channel_image = activation[0, :, :, activation_index]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            ax[row][col].set_axis_off()
            ax[row][col].imshow(channel_image, aspect='auto', cmap='viridis')
            activation_index += 1


# visualize layer 4

for val in [1, 2, 4, 6, 8]:
    display_activation(activations, 8, 8, val)
    plt.show()
