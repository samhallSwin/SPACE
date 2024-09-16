import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time

# Load BigEarthNet using builder
def load_bigearthnet():
    builder = tfds.builder('eurosat/all')
    builder.download_and_prepare()

    # Load 70% of training data and the full test data
    ds_train = builder.as_dataset(split='train[:70%]')
    ds_test = builder.as_dataset(split='train')

    return ds_train, ds_test, builder.info

# Preprocess the images for ResNet50
def preprocess_image(data):
    image = data['sentinel2']
    label = data['label']

    # Resize to 224x224 for ResNet50
    image = tf.image.resize(image, [64, 64])

    # Normalize the image
    image = image / 255.0

    return image, label

# Build the ResNet50 model
def build_resnet50_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
      weights=None, include_top=False, input_shape=(64, 64, 13))

    # Add custom layers for BigEarthNet
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)   # Single-label classification

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Single-label classification loss
                  metrics=['accuracy'])

    return model

# Define a custom callback to track time per epoch
class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_times = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        average_epoch_time = np.mean(self.epoch_times)
        print(f"\nTotal training time: {total_time:.2f} seconds")
        print(f"Average epoch time: {average_epoch_time:.2f} seconds")

# Load, preprocess, and batch the dataset
ds_train, ds_test, info = load_bigearthnet()
num_classes = info.features['label'].num_classes

ds_train = ds_train.map(preprocess_image).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(preprocess_image).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Build and train the ResNet50 model
model = build_resnet50_model(num_classes)

# Create an instance of the time tracking callback
time_callback = TimeHistory()

# Train the model with time tracking
model.fit(ds_train, epochs=5, callbacks=[time_callback])

# Evaluate the model
loss, accuracy = model.evaluate(ds_test)
print(f'Test accuracy: {accuracy}')