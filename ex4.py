# %% Imports
import os
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# %% Functions to Load MNIST from IDX Format
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images.astype('float32') / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# %% Load Local MNIST Dataset
data_path = r'mnist'  #Your local path here

x_train = load_mnist_images(os.path.join(data_path, 'train-images.idx3-ubyte'))
y_train = load_mnist_labels(os.path.join(data_path, 'train-labels.idx1-ubyte'))
x_test = load_mnist_images(os.path.join(data_path, 't10k-images.idx3-ubyte'))
y_test = load_mnist_labels(os.path.join(data_path, 't10k-labels.idx1-ubyte'))

# %% One-hot Encode Labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %% Build the MLP Model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %% Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %% Train the Model
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

# %% Evaluate on Test Set
loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc * 100:.2f}%')
