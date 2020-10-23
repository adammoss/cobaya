sudo apt update && sudo apt install gcc gfortran g++ openmpi-bin openmpi-common libopenmpi-dev libopenblas-base liblapack3 liblapack-dev xorg openbox libnss3 libgtk2.0 libasound2 libxss1
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda create -q -n cobaya-env python=3.7 scipy matplotlib cython PyYAML pytest pytest-forked flaky
source activate cobaya-env
# Install ifort
# Add to .bashrc
#source /opt/intel/bin/compilervars.sh intel64
#export PATH="$HOME/miniconda/bin:$PATH"
#source activate cobaya-env
# Install cobaya
# cd ~
# pip install mpi4py
# pip install cobaya
# cobaya-install cosmo --packages-path cobaya_packages
# Install modified CAMB and cobaya
# pip uninstall cobaya
# cd cobaya_packages/code
# mv CAMB CAMB_old
# git clone https://github.com/adammoss/CAMB
# cd CAMB
# cp -r ../CAMB_old/forutils .
# git checkout spikes
# python setup.py clean
# python setup.py make
# cd ~
# git clone https://github.com/adammoss/cobaya
# cd cobaya
# git checkout spikes
# pip install -e .