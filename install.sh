# exit when any command fails
set -e
# cleanup env and external from previous install
rm -rf env || true
rm -rf external || true

# 1. create virtual env
python3 -m venv env

# 2. install tensorflow
source env/bin/activate
mkdir external

if [ "$1" != "" ]; then
  if [ "$1" == "using_gpu" ]; then
    echo "-----------> installing using tensorflow-gpu"
    pip install tensorflow-gpu==1.13.1
  elif [ "$1" == "using_115" ]; then
    cd external
    pip install gdown
    gdown "https://drive.google.com/uc?id=1qUjg3kfdlK11X3mwVWsBKe4sYuAs-cqU"
    pip install tensorflow-1.15.3-cp36-cp36m-linux_x86_64.whl
    cd ..
  else
    echo "provided argument not recognized"
    exit 1
  fi
else
    echo "installing using tensorflow-cpu"
    pip install tensorflow==1.13.1
fi
pip install tensorflow_probability==0.6.0
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
