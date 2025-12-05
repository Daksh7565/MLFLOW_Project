from pipelines.hybrid_cnn import pipe

if __name__ == "__main__":
    # Define the paths for your data
    # IMPORTANT: Create a 'data' folder and put your image class folders inside it.
    INPUT_DATA_PATH = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\Mlops3\Main dataset"
    OUTPUT_DATA_PATH = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\Mlops3\output" # A folder to store the split data
    pipe(INPUT_DATA_PATH,OUTPUT_DATA_PATH)