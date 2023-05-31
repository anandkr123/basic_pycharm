import os
import numpy as np
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Parameters
cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)

train_x = []
train_y = []
IMAGE_DIR = 'data/train/'
img_height = 64
img_width = 128
batch_size = 256
epochs = 55

alphabets = u"0123456789' "
max_str_len = 5  # max length of input labels
num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank
num_of_timestamps = 16  # max length of predicted labels

# For GPU usage
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def label_to_num(label):
    """
    An index array of character labels
    :param label: Character labels
    :return: Numpy array of indexes of labels
    """
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))

    return np.array(label_num)


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


# List all train images
image_files = os.listdir(IMAGE_DIR)
train_size = len(image_files)

# Get train labels from the file name
for i, image_file in enumerate(image_files):
    train_x.append(image_file)
    train_y.append(label_to_num(str.split(image_file, sep='.')[0]))


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


# Parameter inputs for training
train_x = np.array(train_x)
train_y = np.array(train_y)
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)  # Length of each predicted label.
train_label_len = np.zeros([train_size, 1])  # Length of each true label which is 5.

for i in range(train_size):
    train_label_len[i] = max_str_len

train_output = np.zeros([train_size])  # Output for ctc loss.


# A generator function to read chunks of images
def train_data_generator(train_x, train_y, train_input_len, train_label_len, train_output, batch_size=batch_size):
    while True:

        for i in range(len(train_x) // batch_size):
            yield ([read_batch_images(batch_size, train_x[i * batch_size: (i + 1) * batch_size]),
                    train_y[i * batch_size: (i + 1) * batch_size],
                    train_input_len[i * batch_size: (i + 1) * batch_size],
                    train_label_len[i * batch_size: (i + 1) * batch_size]],

                   train_output[i * batch_size: (i + 1) * batch_size])


# A generator expression
gen = train_data_generator(train_x, train_y, train_input_len, train_label_len, train_output, batch_size=batch_size)


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


# Keras model subclassing for a Resblock
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


# Build the Deep learning OCR Model
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

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # Take predictions onwards 2 since the first couple outputs of the LSTM
        # tend to be garbage
        y_pred = y_pred[:, 2:, :]
        return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    labels = keras.Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
    input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
    label_length = keras.Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    model_final = tf.keras.Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

    return model, model_final


# List of callbacks
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model/dr_model.h5',
                                       monitor="loss",
                                       save_freq='epoch',
                                       save_best_only=True
                                       ),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='loss'),
    tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq='epoch')
]

with tf.device("/gpu:0"):
    model, model_final = build_model()

    model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

    model_final.fit(gen, epochs=epochs,
                    steps_per_epoch=train_size // batch_size,
                    callbacks=my_callbacks)

print("--------TRAINING FINISHED----------\n")


### Test ###
print("--------TESTING ----------\n")



print("-------- READING WEIGHTS----------\n")
model.load_weights('model/dr_model.h5')

print("-------- LOADING IMAGES----------\n")
valid_x = read_batch_images(batch_size=4, image_path=train_x[89:93])
valid_y = train_y[89:93]

preds = model.predict(valid_x)
decoded = tf.keras.backend.get_value(
    tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

print(f'Decoded text is {decoded}\n')

prediction = []
for i in range(len(valid_x)):
    prediction.append(num_to_label(decoded[i]))

print("-------- PREDICTIONS----------\n")
print(f'True label is {valid_y}')
print(f'Predicted label is {prediction}')
