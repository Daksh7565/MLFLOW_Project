from typing import Annotated
import mlflow
from typing import Tuple
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def cnn_trainer(
    data_path: str, epochs: int = 20
) -> Tuple[
    Annotated[tf.keras.Model, "model"],
    Annotated[tf.keras.callbacks.History, "history"],
]:
    """Trains a CNN model and logs the process using MLflow."""
    # CORRECT: Use the TensorFlow autologger for a TensorFlow model.

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_datagen = ImageDataGenerator(
        rescale=(1./255),
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=(1./255))

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(180, 180),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical',
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(180, 180),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical',
    )

    inputs = tf.keras.Input(shape=(180, 180, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    param={"loss":'categorical_crossentropy',
        "optimizer":"adam",
        "metrics":['accuracy']}
    model.compile(**param)
    mlflow.log_params(param)
    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    print("Model training finished.")
    return model, history