# Detic

This repository contains a Python script that performs object detection on a set of images using the Detectron2 model and visualizes the results. The script imports several libraries such as Torch, Detectron2, and Open3D, and defines several functions to load the data, initialize the vocabulary, and perform object detection on the images.

## Requirements
To run this script, you will need:

Python 3.6 or later
Torch 1.7.1 or later
Detectron2 0.5 or later
Open3D 0.13.0 or later
You can install these dependencies using pip:
```
pip install torch detectron2 open3d
```
## Usage
To use this script, follow these steps:

Clone this repository to your local machine.
Navigate to the repository directory.
Download the data file (230321_examples.pkl) and place it in the repository directory.
[Download Here.](https://drive.google.com/file/d/1CjmrPrSXXSQkUilzRLq1rspkvVUaq6DF/view)

Change the DATA_DIR in main.py file.

Run the script using 
```
python3 main.py
```