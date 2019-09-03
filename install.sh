# exit when any command fails
set -e
# 1. install prereqs if not installed

# 2. create virtual env
python3 -m venv env
source env/bin/activate
# 3. install hate-detector
# pip3 install tensorflow-gpu
pip3 install -e .
deactivate
