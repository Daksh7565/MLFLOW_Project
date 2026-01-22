import os
from pipelines.hybrid_cnn import pipe

if __name__ == "__main__":
    # Docker Path Strategy:
    # We will map your actual Windows folders to these specific locations inside the container.
    
    # 1. The input data will be mounted here
    INPUT_DATA_PATH = "./Main dataset"
    
    # 2. The output results will be mounted here
    OUTPUT_DATA_PATH = "./output"
    
    # Ensure output directory exists inside container
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    print(f"Starting pipeline...")
    print(f"Reading data from: {INPUT_DATA_PATH}")
    print(f"Saving output to: {OUTPUT_DATA_PATH}")
    
    pipe(INPUT_DATA_PATH, OUTPUT_DATA_PATH)