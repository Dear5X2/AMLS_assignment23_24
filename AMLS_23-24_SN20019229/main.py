import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Function to run a notebook
def run_notebook(notebook_directory, notebook_name):
    # Construct the full path to the notebook
    notebook_path = os.path.join(notebook_directory, notebook_name)
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as file:
        nb = nbformat.read(file, as_version=4)
        
        # Create an execute preprocessor instance
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': notebook_directory}})
        
        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as file:
            nbformat.write(nb, file)

if __name__ == "__main__":
    # Paths to the notebooks
    pneumonia_notebook_directory = 'A'
    pneumonia_notebook_name = 'PneumoniaMNIST.ipynb'
    
    path_notebook_directory = 'B'
    path_notebook_name = 'PathMNIST.ipynb'
    
    # Run the notebooks
    run_notebook(pneumonia_notebook_directory, pneumonia_notebook_name)
    
    run_notebook(path_notebook_directory, path_notebook_name)
