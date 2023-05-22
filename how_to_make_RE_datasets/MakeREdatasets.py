import numpy as np
import tensorflow as tf
from tensorflow import keras

def randomerasing(image, prob = 0.5, sl = 0.02, sh = 0.4, rl = 0.3):

    if np.random.rand() > prob:
        return image
    
    h, w, _ = image.shape
    area = h*w

    while True:
        erase_area = np.random.uniform(sl, sh)*area
        aspect_ratio = np.random.uniform(rl, 1/rl)

        h_erase = int(np.sqrt(erase_area * aspect_ratio))
        w_erase = int(np.sqrt(erase_area / aspect_ratio))

        left = np.random.randint(0, w)
        top = np.random.randint(0, h)

        if left + w_erase <= w and top + h_erase <= h:
            image[top:top+h_erase, left:left+w_erase, :] = np.random.rand(h_erase, w_erase, 3) * 255.0
            break
    
    return image

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train[:10]

x_train_erased = np.copy(x_train)

for i in range(len(x_train)):
    x_train_erased[i] = randomerasing(x_train[i])