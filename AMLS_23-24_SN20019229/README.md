# AMLS_assignment23_24

##Overview
This project involves two datasets: the PneumoniaMNIST.npz and the PathMNIST.npz. 
The tasks are: A- Binary classify the input from PneumoniaMNIST.npz
               B- Multi-class classify the input from PathMNIST.npz

##Folder Structure
-**A & B**: Involves Jupyter notebooks used for model training, eveluation and visualization (also pre-trained model and original python file)
-**Datasets**: Involves datasets for training, validation, and testing (the actual npz files are not saved, according to requirements)

##Files
-**PneumoniaMNIST.ipynb**: code for training the model of Pneumonia dataset
-**PathMNIST.ipynb**: code for training the model of Path dataset
-**main.py**: Python script to run the trained models and classify images

##Pretrained Models
Pretrained models has been saved in folders

##Usage
To run this project:
1. Load 'PneumoniaMNIST.npz' and 'PathMNIST.npz' and use them to train the models
2. Use 'main.py' to run the saved model, and evalute the performance.

##Package Required
-pandas
-numpy
-matplot.lib
-tensorflow
-sklearn
-cv2
-seaborn
