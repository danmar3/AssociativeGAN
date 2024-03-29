# exit when any command fails
set -e

# prereqs
# sudo apt install build-essential python3-dev python3-venv

# cleanup env and external from previous install
rm -rf env || true
rm -rf external || true

# 1. create virtual env
python3 -m venv env

# 2. install tensorflow
source env/bin/activate
pip install --upgrade pip setuptools
mkdir external

OS_DISTRO=$(lsb_release -si)

if [ "$1" != "" ]; then
  if [ "$1" == "using_113" ]; then
    echo "installing using tensorflow-cpu"
    pip install tensorflow==1.13.1
  elif [ "$1" == "using_113gpu" ]; then
    echo "-----------> installing using tensorflow-gpu"
    pip install tensorflow-gpu==1.13.1
  elif [ "$1" == "using_115" ]; then
    cd external
    pip install gdown
    if [ "$OS_DISTRO" == "Ubuntu" ]; then
      echo "Found Ubuntu Distro..."
      gdown "https://drive.google.com/uc?id=1qUjg3kfdlK11X3mwVWsBKe4sYuAs-cqU"
      pip install tensorflow-1.15.3-cp36-cp36m-linux_x86_64.whl
    elif [ "$OS_DISTRO" == "CentOS" ]; then
      echo "Found CentOS Distro..."
      gdown "https://drive.google.com/uc?id=1mWQOT7trTKlhoQHFWN-R_heTVkEOUUle"
      pip install tensorflow-1.15.3-cp37-cp37m-linux_x86_64.whl
    fi
    cd ..
  elif [ "$1" == "using_py38" ]; then
    cd external
    pip install gdown
    echo "Installing tensorflow 1.15.3 for Python 3.8..."
    gdown "https://drive.google.com/uc?id=1i3W1n8MkIlfIVFShvH2rtHfGnkBA8bpk"
    pip install tensorflow-1.15.3-cp38-cp38-linux_x86_64.whl
    cd ..
  else
    echo "provided argument not recognized"
    exit 1
  fi
else
    echo "installing tensorflow 1.15.3 with cuda 10.1"
    cd external
    pip install gdown
    if [ "$OS_DISTRO" == "Ubuntu" ]; then
      echo "Found Ubuntu Distro..."
      gdown "https://drive.google.com/uc?id=1qUjg3kfdlK11X3mwVWsBKe4sYuAs-cqU"
      pip install tensorflow-1.15.3-cp36-cp36m-linux_x86_64.whl
    elif [ "$OS_DISTRO" == "CentOS" ]; then
      echo "Found CentOS Distro..."
      gdown "https://drive.google.com/uc?id=1mWQOT7trTKlhoQHFWN-R_heTVkEOUUle"
      pip install tensorflow-1.15.3-cp37-cp37m-linux_x86_64.whl
    fi
    cd ..
fi

deactivate

# 1. install prereqs if not installed
source env/bin/activate
# twodlearn
cd external
git clone https://github.com/danmar3/twodlearn.git twodlearn
cd twodlearn
git checkout v0.6
pip install -e .
cd ../
## progressive growing of gans
#git clone https://github.com/tkarras/progressive_growing_of_gans.git
#cd progressive_growing_of_gans
#pip install -r requirements-pip.txt
#cd ../
# go back to root
cd ../
deactivate

# 3. install project
source env/bin/activate
pip3 install -e .
deactivate


# 4. Extras
# source env/bin/activate
# pip install jupyterlab
# jupyter nbextension enable --py widgetsnbextension
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
# pip install voila
# # voila --enable_nbextensions=True
# deactivate
