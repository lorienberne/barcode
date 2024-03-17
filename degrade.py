import os
import random
from PIL import Image, ImageFilter
import numpy as np

def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 0.05, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 1)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

def rotate_image(image):
    return image.rotate(random.uniform(-10, 10))

def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

def degrade_image(image):
    image = add_noise(image)
    image = rotate_image(image)
    image = blur_image(image)
    return image

def load_and_degrade_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            degraded_image = degrade_image(image)
            degraded_image.save(os.path.join(directory, "degraded_" + filename))

load_and_degrade_images("/home/lorienberne/workspace/barcode")