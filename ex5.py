import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        data = batch[b'data']
        labels = np.array(batch[b'labels'])
        return data, labels

def load_cifar10_data(base_path):
    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(base_path, f'data_batch_{i}'))
        x_train.append(data)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = load_cifar10_batch(os.path.join(base_path, 'test_batch'))

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

# Load from local path
x_train, y_train, x_test, y_test = load_cifar10_data(r'C:\Users\lewin\Desktop\Sem6\PRML\PRML Lab\cifar-10-python(1)\cifar-10-batches-py')

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(3072,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)
# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc * 100}%')



