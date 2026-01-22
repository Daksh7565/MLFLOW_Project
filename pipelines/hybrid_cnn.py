
from steps.data import data_splitter
from steps.model import cnn_trainer
from steps.evalute import evaluator
from steps.Model2 import Hybrid_model
import logging
import mlflow 
import os 

def pipe(data_path:str, output_path:str):
    # 1. Enable System Metrics (CPU/GPU/RAM)
    # This will log metrics every 10 seconds by default
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    
    mlflow.set_experiment("Stark")
    
    # Start the run here so everything inside is tracked under one ID
    with mlflow.start_run(run_name="Hybrid_Training_Run"):
        
        split_data_path = data_splitter(data_path, output_path)
        
        # Train Basic CNN
        model, history = cnn_trainer(split_data_path)
        
        # Train Hybrid Model
        model2, history2 = Hybrid_model(model=model, path=split_data_path)
        
        # Evaluate
        evaluator(model=model2, history=history2, data_path=split_data_path)