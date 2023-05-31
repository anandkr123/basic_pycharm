import os
import glob
import sys

import cv2
# import matplotlib.pyplot as plt
import pathlib
import imageio.v3 as im
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)
IMAGE_DIR = 'data/train/'
img_height = 64
img_width = 128


def load_data(input_dir):
    """
    :return: complete paths of images in input_dir
    """

    input_images_path = (glob.glob(os.path.join(cwd, input_dir, '*.*')))

    try:
        return input_images_path
    except FileNotFoundError:
        print("Oops! ")


# total_input_path = load_data(input_dir=join_dir(cwd, INPUT_DIR, 'train'))

# def read_image(*args, minimum=0):
#     """
#     reads the input image and normalizes the input image in range 0 and 1
#     :return: image
#     """
#     # img = Image.open(join_dir(*args))
#     cv2.
#     img = cv2.imread(join_dir(*args), 0)
#     norm_image = cv2.normalize(img, None, alpha=minimum, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)
#     img1 = np.expand_dims(norm_image, axis=-1)
#     return img1
#
#
# def read_batch_images(batch_size, image_path, img_ht, img_wt):
#     """
#     read images from the list of image paths provided
#
#     :param batch size: number of images to read
#     :param image_path : list of image paths
#     :return: read images in numpy array
#
#     """
#
#     counter = 0
#     # if total  images// batch size does not fit , it takes the zeros into array
#     if not image_path:
#         print("There are no more images ! Check the image paths, Must generate error !")
#         return
#     if len(image_path) != batch_size:
#         print("Not enough images in the image_path . will add zero value(black) images in addition")
#     images = np.zeros((batch_size, img_ht, img_wt, 1), dtype=np.float32)
#     for path in image_path:
#         img = read_image(path, minimum=0)
#         if img.shape[0] != img_ht or img.shape[1] != img_wt:
#             print("Check image dimensions in directory and default args value in img_ht and img_wt")
#             exit()
#         images[counter] = img
#         counter += 1
#     return images
#
#
# def train_data_generator(input_path, batch_size=train_batch_size):
#     while True:
#
#         input_path,  = shuffle(input_path, random_state=randrange(100))
#
#         for i, j in zip(range(len(input_path) // batch_size), range(len(ground_path) // batch_size)):
#             yield (read_batch_images(batch_size, input_path[i * batch_size: (i + 1) * batch_size], img_ht, img_wt),
#                    read_ground_batch_images(batch_size, ground_path[j * batch_size: (j + 1) * batch_size], img_ht,
#                                             img_wt))

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(ModelConfig.vocab.find(ch))
    return np.array(label_num)


def process_path(file_path):
    number_of_timesteps = 16
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, dtype=tf.float32)  #
    image = image/255.0
    label = tf.strings.split(file_path, '/')[-1]
    label = tf.strings.split(label, '.')[0]
    label = tf.strings.bytes_split(label)
    label = tf.strings.to_number(label, tf.float32)

    train_label_length = tf.constant(ModelConfig.max_text_length)
    train_input_length = tf.constant(number_of_timesteps-2)

    train_output = np.zeros(1)

    # label = tf.pad(label, [[0, 11]])
    return image, label, train_input_length, train_label_length


class ModelConfig:
    model_path = os.path.join('Models/1_image_to_word', datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
    vocab = u"0123456789-' "
    height = 64
    width = 128
    max_text_length = 5
    batch_size = 64
    learning_rate = 1e-4
    train_epochs = 100
    train_workers = 20


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, training=False):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same", )
        self.bn = layers.BatchNormalization(trainable=training)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, channels, training=False):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.cnn1 = CNNBlock(channels[0], 3, training=training)
        self.cnn2 = CNNBlock(channels[1], 3, training=training)
        self.cnn3 = CNNBlock(channels[2], 3, training=training)
        # self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], kernel_size=1, trainable=training, padding="same")

    def call(self, inputs, *args, **kwargs):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.cnn3(x + self.identity_mapping(inputs))
        # print(x.shape)
        # x = self.pooling(x)
        return x


# class ResNetModel(tf.keras.Model):
#     def __init__(self, output_dim):
#         super(ResNetModel, self).__init__()
#         # self.input_dim = input_dim
#         self.res_block1 = ResBlock([32, 32, 64])
#         self.res_block2 = ResBlock([64, 64, 128])
#         self.blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
#         # self.pool = tf.keras.layers.GlobalAvgPool2D()
#         self.final = tf.keras.layers.Dense(output_dim + 1)
#
#     def call(self, inputs, training=False, mask=None):
#         x = self.res_block1(inputs, training=True)
#         x = self.res_block2(x, training=True)
#         # x = self.pool(x)
#         x = keras.layers.Reshape((x.shape[-2] * x.shape[-3], -1))
#         # x = self.squeeze(x)
#         x = self.blstm(x, trainable=True)
#         x = self.final(x, activation='softmax', name='final', trainable=True)
#         return x

    # def model(self):
    #     x = keras.Input()   # specify input shape
    #     return keras.Model(inputs=[x], outputs=self.call(x))


def ctc_loss(args):
    """Compute the CTC loss between y_true and y_pred."""

    y_true, y_pred = args
    y_pred = y_pred[:, 2:, :]

    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    # label_length = tf.constant(16, dtype='int64')
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true=y_true, y_pred=y_pred, input_length=input_length,
                                           label_length=label_length)

    return loss


ds_train = tf.data.Dataset.list_files(str(pathlib.Path(IMAGE_DIR+'*.png')))

ds_train = ds_train.map(process_path).batch(batch_size=ModelConfig.batch_size)


inputs = keras.Input(shape=(64, 128, 1))
layer_resblock_1 = ResBlock([32, 32, 64])(inputs)
layer_max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(layer_resblock_1)

layer_resblock_2 = ResBlock([64, 64, 128])(layer_max_pool_1)
layer_max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(layer_resblock_2)
new_shape = ((ModelConfig.height//4), (ModelConfig.width//4) * 128)

layer_squeezed = layers.Reshape(target_shape=new_shape, name="reshape")(layer_max_pool_2)
x = layers.Dense(16, activation="relu", name="dense1")(layer_squeezed)
layer_blstm = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

output = layers.Dense(len(ModelConfig.vocab)+1, activation='softmax', name="output")(layer_blstm)

model = keras.models.Model(inputs=inputs, outputs=output)


labels = keras.Input(name='gtruth_labels', shape=[ModelConfig.max_text_length], dtype='float32')
input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
label_length = keras.Input(name='label_length', shape=[1], dtype='int64')

ctc_loss_function = layers.Lambda(ctc_loss, output_shape=(1,), name='ctc')([labels, output])

model_final = keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss_function)


model_final.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=ModelConfig.learning_rate),
    loss={'ctc': lambda y_true, y_pred: y_pred},
    metrics=['accuracy']
)

train_output = tf.zeros(shape=ModelConfig.batch_size)


model_final.fit(zip(ds_train, train_output), verbose=1, epochs=2)


image = im.imread(join_dir(cwd, IMAGE_DIR, '49621.png'))
image = image/255.0

image_array = np.asarray(image)

# Expand the dimensions of the array
image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
image_array = np.expand_dims(image_array, axis=-1)


pred = model.predict(image_array)
print(f'the predicted shape is {pred.shape}')

decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1],
                                   greedy=True)[0][0])

digit_recognized = ("".join(str(e) for e in decoded[0]))

print('\nDigit_Recognized:\n' + digit_recognized)