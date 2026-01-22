
from steps.data import data_splitter
from steps.model import cnn_trainer
from steps.evalute import evaluator
import logging
import mlflow 
def pipe(data_path:str,output_path:str):
    # This step returns the path to the split
    mlflow.set_experiment("Stark")
    split_data_path = data_splitter(data_path, output_path)
    
    # This step now correctly returns placeholders for the model and history
    model, history = cnn_trainer(split_data_path)
    
    # Pass the model, history, and data path to the evaluator step
    # The logic that caused the error is now removed from here
    evaluator(model=model, history=history, data_path=split_data_path)