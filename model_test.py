import os
import sys

import numpy as np
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def read_image(image_file):
    """
    Read an image from a path
    :param image_file: File path
    :return: Returns a  normalised image
    """
    img_dir = join_dir(cwd, IMAGE_DIR, image_file)
    image = imageio.v3.imread(img_dir)
    image = image / 255.
    return image


def num_to_label(num):
    """
    Transform predictions to original characters.
    :param num: Array of indexes.
    :return: A string with the original label.
    """
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret


def read_batch_images(batch_size, image_path, img_ht=64, img_wt=128):
    """
    read images from the list of image paths provided

    :param batch size: number of images to read
    :param image_path : list of image paths
    :return: read images in numpy array

    """

    counter = 0
    if len(image_path) != batch_size:
        print("Not enough images in the image_path . will add zero value(black) images in addition")
    images = np.zeros((batch_size, img_ht, img_wt), dtype=np.float32)
    for path in image_path:
        img = read_image(path)
        if img.shape[0] != img_ht or img.shape[1] != img_wt:
            print("Check image dimensions in directory and default args value in img_ht and img_wt")
            exit()
        images[counter] = img
        counter += 1
    return images


# Keras model subclassing for Conv block
class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, training=False):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same", name='conv',
                                  kernel_initializer='he_normal', trainable=training)
        self.bn = layers.BatchNormalization(trainable=training)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


# Keras model subclassing for Res block
class ResBlock(layers.Layer):
    def __init__(self, channels, training=False):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.cnn1 = CNNBlock(channels[0], 3, training=training)
        self.cnn2 = CNNBlock(channels[1], 3, training=training)
        self.cnn3 = CNNBlock(channels[2], 3, training=training)
        self.pooling = layers.MaxPooling2D(name='max_pooling')
        self.identity_mapping = layers.Conv2D(channels[1], kernel_size=1, trainable=training, padding="same",
                                              name='identity')

    def call(self, inputs, *args, **kwargs):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.cnn3(x + self.identity_mapping(inputs))
        x = self.pooling(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


# Model for label predictions
def build_model():
    input_data = keras.Input(shape=(img_height, img_width, 1), name='input')
    inner = ResBlock([32, 32, 64], training=True)(input_data)

    inner = ResBlock([64, 64, 128], training=True)(inner)
    inner = layers.Dropout(0.3)(inner)

    # CNN to RNN

    inner = layers.Reshape(target_shape=(16, 32 * 128), name='reshape')(inner)
    inner = layers.Dense(16, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

    # RNN
    inner = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name='lstm1')(inner)
    inner = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='lstm2')(inner)

    # OUTPUT
    inner = layers.Dense(num_of_characters, kernel_initializer='he_normal', name='dense2')(inner)
    y_pred = layers.Activation('softmax', name='softmax')(inner)

    model = tf.keras.Model(inputs=input_data, outputs=y_pred)

    return model


# Parameters
cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)
IMAGE_DIR = 'data/test/'
test_y = []

img_height = 64
img_width = 128

alphabets = u"0123456789' "
num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank


test_images_paths = os.listdir(IMAGE_DIR)
test_size = len(test_images_paths)

for i, image_file in enumerate(test_images_paths):
    test_y.append((str.split(image_file, sep='.')[0]))

test_images = read_batch_images(batch_size=60, image_path=test_images_paths)
img_new = test_images[3]


print("======= Build model ==========\n")
model = build_model()

print("=========Load weights==========\n")
model.load_weights('model/dr_model.h5')

# Model predictions and decoded
preds = model.predict(test_images[3:4])

# print(preds)

decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1],
                                    greedy=True)[0][0])
digit_recognized = ("".join(str(e) for e in decoded[0]))

print(decoded)
print(digit_recognized)
sys.exit()

print("=========Predictions==========\n")
prediction = []
for i in range(test_size):
    prediction.append(num_to_label(decoded[i]))


y_true = test_y

correct_char = 0
total_char = 0
correct = 0

for i in range(test_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)

    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1


print("=========Character prediction accuracy==========\n")
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))

print("\n=========Sample True Vs Predictions==========\n")

for i in range(10):
    print(f'True label\t {y_true[i]}:\t Predicted label\t{prediction[i]}')