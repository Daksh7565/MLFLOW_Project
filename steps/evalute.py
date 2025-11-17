# import logging
# import mlflow
# from mlflow.models import infer_signature
# from typing import Annotated
# import tensorflow as tf
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
# import numpy as np # Import numpy


# def evaluator(
#     model: tf.keras.Model,
#     history: tf.keras.callbacks.History,
#     data_path: str,) -> Annotated[float, "test_accuracy"]:
#     """
#     Evaluates the model on the test set and logs metrics from training history.
    
#     NOTE: This function assumes it is being called within an active MLflow run.
#     """
#     print("--- Starting Evaluation Step ---")

#     history_df = pd.DataFrame(history.history)
#     print("Full training history:")
#     print(history_df)
#     logging.info(f"The training history is:\n{history_df}")

#     # --- CORRECTED METRIC LOGGING ---
#     # Log the training/validation history for each epoch
#     for epoch, row in history_df.iterrows():
#         # The 'step' parameter tells MLflow this is a time-series metric
#         mlflow.log_metric("epoch_accuracy", row["accuracy"], step=epoch)
#         mlflow.log_metric("epoch_loss", row["loss"], step=epoch)
#         mlflow.log_metric("epoch_val_accuracy", row["val_accuracy"], step=epoch)
#         mlflow.log_metric("epoch_val_loss", row["val_loss"], step=epoch)

#     # Log the training history dataframe as a CSV artifact
#     # Assuming history_df is your pandas DataFrame 
#     history_csv_path = "training_history.csv" 

#     # Save the DataFrame to a CSV file 
#     history_df.to_csv(history_csv_path, index=False) 

#     # Log the CSV file as an artifact 
#     mlflow.log_artifact(history_csv_path)
#     print("\nEvaluating model on the test set...")
#     test_dir = os.path.join(data_path, 'test')

#     test_datagen = ImageDataGenerator(rescale=(1./255))
#     test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(180, 180),
#         batch_size=16,
#         color_mode='rgb',
#         class_mode='categorical',
#         shuffle=False
#     )

#     test_loss, test_accuracy = model.evaluate(test_generator)
#     print(f"Test Loss: {test_loss:.4f}")
#     print(f"Test Accuracy: {test_accuracy:.4f}")

#     # Log final test metrics
#     mlflow.log_metric("final_test_accuracy", test_accuracy)
#     mlflow.log_metric("final_test_loss", test_loss)
#     # (Optional but good practice) Log the full report as a text artifact
#     mlflow.log_text(history_df, "training_classification_report.txt")
#     # --- END OF CORRECTIONS ---
#     sample_batch, _ = next(iter(test_generator))
#     # Infer the model signature
#     signature = infer_signature(sample_batch, model.evaluate(sample_batch))

#     # Log the model
#     model_info = mlflow.tensorflow.log_model(
#         tf_model=model,
#         artifact_path="cotten_leaf", # Use artifact_path instead of name
#         signature=signature,
#         input_example=test_datagen,
#         registered_model_name="CNN_V1",
#     )
#     mlflow.set_logged_model_tags(
#         model_info.model_id, {"Training Info": "Basic CNN model for image classification"}
#     )
#     return test_accuracy

import logging
import mlflow
from mlflow.models import infer_signature
from typing import Annotated
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def evaluator(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    data_path: str,
) -> Annotated[float, "test_accuracy"]:
    """
    Evaluates the model on the test set and logs metrics, artifacts, and the model.

    NOTE: This function assumes it is being called within an active MLflow run.
    """
    print("--- Starting Evaluation Step ---")

    history_df = pd.DataFrame(history.history)
    print("Full training history:")
    print(history_df)
    logging.info(f"The training history is:\n{history_df}")

    # --- Log metrics for each epoch ---
    # Log the training/validation history for each epoch
    for epoch, row in history_df.iterrows():
        # The 'step' parameter tells MLflow this is a time-series metric
        mlflow.log_metric("epoch_accuracy", row["accuracy"], step=epoch)
        mlflow.log_metric("epoch_loss", row["loss"], step=epoch)
        mlflow.log_metric("epoch_val_accuracy", row["val_accuracy"], step=epoch)
        mlflow.log_metric("epoch_val_loss", row["val_loss"], step=epoch)

    # --- Log artifacts ---
    # Log the training history dataframe as a CSV artifact
    history_csv_path = "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    mlflow.log_artifact(history_csv_path)

    print("\nEvaluating model on the test set...")
    test_dir = os.path.join(data_path, 'test')

    test_datagen = ImageDataGenerator(rescale=(1./255))
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(180, 180),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Log final test metrics
    mlflow.log_metric("final_test_accuracy", test_accuracy)
    mlflow.log_metric("final_test_loss", test_loss)

    # --- CORRECTED CODE ---
    # Convert the DataFrame to a string before logging it as a text artifact
    report_text = history_df.to_string()
    mlflow.log_text(report_text, "training_classification_report.txt")
    # --- END OF CORRECTION ---

    # --- Log the model ---
    # Infer the model signature
    sample_batch, _ = next(iter(test_generator))
    signature = infer_signature(sample_batch, model.predict(sample_batch))

    # Log the model to MLflow
    model_info = mlflow.tensorflow.log_model(
        model=model,
        signature=signature,
        input_example=sample_batch,
        artifact_path="hybrid_model",  # This will be the folder name in MLflow artifacts
        registered_model_name="hybrid_image_classifier" # This name appears in the MLflow Model Registry
    )
    print(f"Model logged with URI: {model_info.model_uri}")

    return test_accuracy