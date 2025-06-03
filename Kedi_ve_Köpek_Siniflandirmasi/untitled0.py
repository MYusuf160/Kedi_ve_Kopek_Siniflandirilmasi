# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:45:02 2024

@author: Yusuf
"""

# Importing required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Defining image size and batch size
img_size = (150, 150)
batch_size = 32

# Creating ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, #her pikselin değerini 0-1 aralığına dönüştürür. 
    shear_range=0.2, #Görüntülerde rastgele kesme (shear) dönüşümleri uygular.
    zoom_range=0.2, #Görüntülerde rastgele yakınlaştırma (zoom) işlemleri yapar.
    horizontal_flip=True #Görüntüleri yatay olarak rastgele çevirir.
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Creating training and testing sets
training_set = train_datagen.flow_from_directory(
    r'C:\Users\Yusuf\Desktop\Bitirme\training_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r'C:\Users\Yusuf\Desktop\Bitirme\test_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Defining the model architecture with Dropout regularization
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fitting the model
history = model.fit(training_set, epochs=25, validation_data=test_set)

# Save the trained model
model.save("cat_dog_model.h5")

# Load the trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.tight_layout()
plt.show()

# Reset the test set generator
test_set.reset()

# Get the true labels and images from the test set
true_labels = []
images = []

for i in range(len(test_set)):
    img, label = next(test_set)
    images.extend(img)
    true_labels.extend(label)

# Convert images to numpy array
images = np.array(images)

# Make predictions
predictions = model.predict(images)

# Convert predictions to binary labels
predicted_labels = [int(pred > 0.5) for pred in predictions]

# Define the classes
classes = ["cat", "dog"]

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
plt.title('Confusion Matrix')
plt.show()

# Select 10 random images for visualization
random_indices = random.sample(range(len(images)), 10)

# Visualize 10 random predictions
plt.figure(figsize=(20, 10))

for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[idx])
    plt.title(f"Predicted: {classes[predicted_labels[idx]]}\nActual: {classes[int(true_labels[idx])]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
