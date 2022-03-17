#!/bin/bash 
SBATCH --nodes=3                        # requests 3 compute servers
SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
SBATCH --cpus-per-task=1                # uses 1 compute core per task
SBATCH --time=1:00:00
SBATCH --mem=10GB
SBATCH --job-name=python-hello-world
SBATCH --output=python-hello-world.out

module purge
module load python/intel/3.8.6
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

python ./hello-world.py

# unload module
module unload python3/intel/3.6.3