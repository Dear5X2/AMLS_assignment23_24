import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    with open(notebook_path) as file:
        nb = nbformat.read(file, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': './'}})

if __name__ == "__main__":
    run_notebook('PneumoniaMNIST.ipynb')
    run_notebook('PathMNIST.ipynb')

