import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Paths to data directories
train_dir = r'C:\Users\imena\Downloads\FaceMaskDataset.zip\FaceMaskDataset\train224'
test_dir = r'C:\Users\imena\Downloads\FaceMaskDataset.zip\FaceMaskDataset\test224'

# Data augmentation for training and validation
data_augmentation = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = data_augmentation.flow_from_directory(
    train_dir, bnn<n°++++++++++++++++++++++++++++++++++¨BN °+++++++++++++++
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model 1: CNN from scratch
model_scratch = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_scratch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training Model 1
history_scratch = model_scratch.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Model 2: Transfer learning without fine-tuning
base_model_no_ft = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model_no_ft.layers:
    layer.trainable = False

x_no_ft = base_model_no_ft.output
x_no_ft = Flatten()(x_no_ft)
x_no_ft = Dense(256, activation='relu')(x_no_ft)
x_no_ft = Dropout(0.5)(x_no_ft)
output_no_ft = Dense(1, activation='sigmoid')(x_no_ft)

model_transfer_no_ft = Model(inputs=base_model_no_ft.input, outputs=output_no_ft)
model_transfer_no_ft.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training Model 2 without fine-tuning
history_transfer_no_ft = model_transfer_no_ft.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Model 3: Transfer learning with fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model_transfer = Model(inputs=base_model.input, outputs=output)
model_transfer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tuning the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Training Model 3 with fine-tuning
history_transfer = model_transfer.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Plotting results
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history_scratch, 'Model from Scratch Accuracy')
plot_history(history_transfer_no_ft, 'Transfer Learning without Fine-Tuning Accuracy')
plot_history(history_transfer, 'Transfer Learning with Fine-Tuning Accuracy')

# Evaluate all models
print("Model from Scratch Evaluation:", model_scratch.evaluate(val_generator))
print("Transfer Learning without Fine-Tuning Evaluation:", model_transfer_no_ft.evaluate(val_generator))
print("Transfer Learning with Fine-Tuning Evaluation:", model_transfer.evaluate(val_generator))
