# Computer Vision Programming Assignment 1
Programming assignment 1 for computer vision course.
- Super resolution using gradient descent
- Super resolution using blur kernel and modified Rechardson-Lucy deconvolution

## Dependencies
* python version: python v.3.10
* OpenCV: at least v.4.6.0
* numpy: at least v.1.23.0
* tqdm: at least v.4.20.0

## Usage
```bash
# install dependencies
pip install numpy
pip install opencv-python
pip install tqdm

# only if libgl is not installed
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglib2.0-0

# run codes
cd Codes
python3 problem1.py
python3 problem2.py
python3 addtional_method.py
```

## Directories
* Analysis: Stuffs to see hyperparameter tendencies
* Codes: Source code for this project
* Figures: Figures that used for the report
* Images: Given image and project results.

## Components
* Images
    * HR.png: Groud-truth image
    * upsampled.png: Image that is upsampled (blurry)
    * problem1.png: Result image from problem1.py code
    * problem2.png: Result image from problem2.py code
    * additional.png: Result form additional image

* Codes
    * problem1.py: Solution for problem 1
    * problem2.py: Solution for problem 2
    * additional_method.py: My additional method implementation
    * KERNEL.csv: Blur kernel matrix for additional method
