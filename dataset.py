import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
from os import path, getcwd, makedirs
from tqdm import tqdm


def get_subdirectory(sd):
    """
    Create the sub directory to 'data' parent directory
    :param sd: Create sub directory to data if it doesn't exist
    :return: returns the directory
    """
    dir = path.join(getcwd(), f'data/{sd}')
    if not path.isdir(dir):
        makedirs(dir, exist_ok=True)
    return dir


def list_full_paths(directory):
    """
    List absolute path of all the files.
    :param directory: Directory names.
    :return: Absolute path of all the files in the directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)]


# Parameters
NUM_IMAGES = 60         # Number of images to create
IMAGE_SIZE = (128, 64)  # Width x Height
NUM_DIGITS = 5
DIGIT_SIZE = 22         # Width x Height
FONT_SIZE = 30

FONT_DIR = "Fonts"  # Change to your preferred font path
FONT_FILES = list_full_paths((path.join(getcwd(), FONT_DIR)))

# Generate images
images = []
labels = []

for i in tqdm(range(NUM_IMAGES)):
    # Create empty image
    image = Image.new("L", IMAGE_SIZE, color=255)  # Gray scale image

    # Generate digits and labels
    digits = []
    label = ""
    random_font = random.choice(FONT_FILES)
    for j in range(NUM_DIGITS):
        digit = str(random.randint(0, 9))
        digits.append(digit)
        label += digit

    # Draw digits onto image
    font = ImageFont.truetype(font=random_font, size=random.randint(18, 25))
    draw = ImageDraw.Draw(image)
    padding = random.randint(1, 5)
    space = random.randint(8, 15)
    DIGIT_SIZE = random.randint(13, 20)
    for j, digit in enumerate(digits):
        x = j * (DIGIT_SIZE + padding) + space
        y = (IMAGE_SIZE[1] - DIGIT_SIZE) / 2
        draw.text((x, y), digit, font=font, fill=0)
    # Save the image
    image.save(f"{get_subdirectory('test')}/{label}.png", format='png')


