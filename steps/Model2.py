import mlflow
import tensorflow as tf
import logging
from typing import Tuple
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.tensorflow
def Hybrid_model(model:tf.keras.Model,path:str)-> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    # ... (your existing data generator code remains the same) ...
    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, 'val')
    train_datagen = ImageDataGenerator(
    rescale= (1./255),
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=(1./255))

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224,224),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')
        
    # Autolog will still capture metrics, params, etc.

    mobilenet = tf.keras.applications.MobileNet(weights='imagenet', include_top=False)
    # ... (your model building logic remains the same) ...
    resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

    mobilenet_base = tf.keras.Model(inputs=mobilenet.input, outputs=mobilenet.layers[-2].output)
    resnet_base = tf.keras.Model(inputs=resnet.input, outputs=resnet.layers[-2].output)

    new_input = tf.keras.layers.Input(shape=mobilenet.input_shape[1:])

    mobilenet_output = mobilenet_base(new_input)
    resnet_output = resnet_base(new_input)

    mobilenet_output_flatten = tf.keras.layers.GlobalAveragePooling2D()(mobilenet_output)
    resnet_output_flatten = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)

    concat_features = tf.keras.layers.concatenate([mobilenet_output_flatten, resnet_output_flatten])

    existing_model = model
    existing_output = existing_model(new_input)

    combined_output = tf.keras.layers.concatenate([existing_output, concat_features])

    x = tf.keras.layers.Dense(256, activation='relu')(combined_output)
    x = tf.keras.layers.Dense(6, activation='softmax')(x)

    new_model = tf.keras.Model(inputs=new_input, outputs=x)
    # --- ADD THIS SECTION: Log Hybrid Architecture ---
    # 1. Detailed Description
    custom_desc = """
    Hybrid Model Architecture Details:
    ----------------------------------
    Backbone 1: MobileNet (ImageNet weights, top removed)
    Backbone 2: ResNet50 (ImageNet weights, top removed)
    Fusion: Concatenated GlobalAveragePooling outputs
    Head: Dense(256) -> Dense(6, Softmax)
    """
    
    # 2. Capture technical summary
    string_list = []
    new_model.summary(print_fn=lambda x: string_list.append(x))
    full_summary = "\n".join(string_list)
    
    # 3. Log to MLflow
    complete_report = custom_desc + "\n" + "="*30 + "\n" + full_summary
    mlflow.log_text(complete_report, "hybrid_model_architecture.txt")
    # -------------------------------------------------
    param={"loss":'categorical_crossentropy',
        "optimizer":"adam",
        "metrics":['accuracy']}
    new_model.compile(**param)
    mlflow.log_params(param)
    history = new_model.fit(train_generator, epochs=1, validation_data=val_generator)

    # # --- ADD THIS SECTION FOR EXPLICIT MODEL LOGGING ---
    # # Log the model to MLflow with a specific artifact path
    # mlflow.tensorflow.log_model(
    #     model=new_model,
    #     artifact_path="hybrid_model",  # This will be the folder name in MLflow artifacts
    #     registered_model_name="hybrid_image_classifier" # This name appears in the MLflow Model Registry
    # )
    # # ----------------------------------------------------
    
    return new_model,history