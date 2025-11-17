import logging
import pandas as pd
from typing import Annotated
import splitfolders as s

# The IngestData class is defined but not used in the pipeline.
# It can be kept or removed.
class IngestData:
    def __init__(self,data_path: str,output_path :str ):
        self.data_path=data_path
        self.output_path=output_path
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return

# CORRECT: Removed the unnecessary experiment_tracker from this step.
def data_splitter(input_path: str, output_path: str) -> Annotated[str, "output_path"]:
    """Splits image data into train, validation, and test sets."""
    print(f"Splitting data from '{input_path}' into '{output_path}'...")
    s.ratio(input_path, output=output_path, seed=42, ratio=(0.8, 0.1, 0.1))
    print("Data splitting complete.")
    return output_path