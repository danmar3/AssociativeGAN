# AssociativeGAN
start training
```
python3 runit.py --n_steps=1000 --n_steps_save=5 --gpu=0 --dataset="cifar10" --indicator="cifar10" --model=wacgan
```


restore a session
```
python3 runit.py --n_steps=1000 --n_steps_save=5 --gpu=4 --dataset="stl10" --indicator="stl10" --model=wacgan --session="tmp/wacgan/session_20200315_0441_stl10"
```
