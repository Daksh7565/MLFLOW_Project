from pipelines.hybrid_cnn import pipe

if __name__ == "__main__":
    # Define the paths for your data
    # IMPORTANT: Create a 'data' folder and put your image class folders inside it.
    INPUT_DATA_PATH = "C:\\Users\\jaydu\\OneDrive\\Desktop\\python_projects\\Mlops1\\Main dataset"
    OUTPUT_DATA_PATH = "C:\\Users\\jaydu\\OneDrive\\Desktop\\python_projects\\Mlops1\\output" # A folder to store the split data
    pipe(INPUT_DATA_PATH,OUTPUT_DATA_PATH)