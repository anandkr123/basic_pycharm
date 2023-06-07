# OCR digit recognition

Train a deep learning model to read characters from simulated number plates pictures (OCR). In particular the algorithm is supposed
to always read a fixed amount of numbers between [0,9]. The system will be presented with cropped plates
with 5 numbers each. E.g. 37463 or 23837

## Data set creation

Build a full example with artificial generated samples of at least 60.000 images. Each with labels. The amount of numbers between [0,9]. The system will be presented with cropped plates with 5 numbers each.
E.g. 37463 or 23837
The task is to generate artificial image samples and their corresponding labels for an OCR (Optical Character Recognition) Deep learning model. The images will be

- grayscale and will contain 5 digits each,
- Size of (64, 128)
- with each digit being a number between 0 and 9.
- There will be at least 60,000 images generated, each with a different set of 5 digits.
- The images will vary in font type and size, with 22 different font types used and font sizes ranging from 18 to 25.

Additionally, each image will have a different character spacing, adding further variety to the data samples. The goal of introducing these nuances is to create a diverse and realistic data set for training an OCR model that can accurately read and recognize digits in various styles and sizes. Training images can be created
using ‘data_generation.ipynb’ specifying the directory = ‘train’. The images are created in data/train directory.

Sample images from dataset:

![40863](https://github.com/anandkr123/ocr-digit-recognition/assets/23450113/b9da93be-33a6-4dbb-a355-6e68dfbd3774)

![69808](https://github.com/anandkr123/ocr-digit-recognition/assets/23450113/45301612-c114-49d2-9147-c376ca1c56cf)


## Deep learning model

Text recognition pipeline (CNN+LSTM) with CTC loss

![Screenshot from 2023-05-31 10-47-53](https://github.com/anandkr123/ocr-digit-recognition/assets/23450113/03f24589-2ce6-48e9-8d84-247c6c03ad3a)


# (Connectionist Temporal classification) CTC Loss

Connectionist Temporal Classification (CTC) loss is a widely used loss function in Optical Character Recognition (OCR) models. It is specifically designed to handle sequence labeling tasks where the alignment between the input sequence and the target sequence isnot one-to-one.

In OCR, CTC loss is used to train a deep learning model to predict sequences of characters from input images, such as recognizing text from images of handwritten or printed documents. The main advantage of CTC loss is that it does not require explicit alignment between the input and target sequences, making it suitable for tasks where the length of the input and output sequences may differ. In the proposed deep learning model, we have a fixed label length i.e. 5 as the training
samples have only 5 digits. At every time step, probabilities are calculated over the complete vocabulary (“0123456789' “).

Parameters used in CTC loss for an OCR model typically include:
1. Input probabilities: The output probabilities of the neural network for each time step. These probabilities represent the likelihood of each character or blank symbol occurring at each time step.
2. Target labels: The ground truth labels or target sequence of characters for the input image. It consists of the characters (digits) in the correct order in an array.
3. Input sequence length: The length of the input sequence or the number of time steps in the output probabilities. It determines the number of steps needed for decoding and aligning the predicted and target sequences.
4. Target sequence length: The length of the target sequence or the number of characters in the ground truth label. It provides information about the number of characters in the input image.
5. Blank symbol: The blank symbol is used to account for repeated characters or gaps in the predicted sequence. It helps to handle situations where there are multiple characters in the input image, but fewer characters are required in the target label.

During training, CTC loss calculates the likelihood of the target sequence given the input probabilities, taking into account all possible alignments. The loss function maximizes the probability of the correct label sequence while accounting for possible deletions, insertions, and
repetitions of characters. By optimizing the CTC loss, the OCR model can effectively learn to recognize and transcribe text from images.

# Model training

The model is trained over google colab, To get an overview of training and predictions, please follow this Google colab [notebook](https://colab.research.google.com/drive/1UP9YlU0v8u2i7nnDH6UMFw_tm8nV7ozP?authuser=2#scrollTo=j8NcITX_6rda)
- The optimizer used is Adam with a learning rate of 3e-4.
- For model training, we are tracking the training loss with patience of 5.
- Number of epochs = 55, batch_size = 256

Below figure shows the training loss over epoch

![Screenshot from 2023-05-16 12-08-11](https://github.com/anandkr123/ocr-digit-recognition/assets/23450113/e8016eff-38ff-4ce2-a6fe-0751b76d0a5a)


Due to patience = 5, the training stopped at 43rd epoch. The loss reached a minimum value
of 0 .0015

# Predcitions

Below are some predicted labels over the test set.
Thetest set contains 60 images in (‘data/test’ directory) generated in a similar way as for training. The script restore the weights of the trained model dr_model.h5. The below images shows the results after running the **script ‘model_test.py’** .

![Screenshot from 2023-05-16 12-15-36](https://github.com/anandkr123/ocr-digit-recognition/assets/23450113/5deef8de-5ba6-4362-adb3-c3184c10c1ec)


