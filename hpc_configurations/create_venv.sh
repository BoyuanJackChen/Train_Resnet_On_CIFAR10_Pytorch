#!/bin/bash

# activate existed python3 module to get virtualenv
module load python3/intel/3.6.3

# create virtual environment with python3
virtualenv -p python3 .env

# activate virtual environment
source .env/bin/activate

pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install tensorflow-gpu
pip install jupyter
# uncomment the following line if you have requirements.txt file
# pip install -r requirements.txt

# unload module
module unload python3/intel/3.6.3