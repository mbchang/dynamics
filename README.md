# [Neural Physics Engine](http://mbchang.github.io/npe)

This repository contains the code described in [https://arxiv.org/abs/1612.00341](https://arxiv.org/abs/1612.00341).
Project website: [http://mbchang.github.io/npe](http://mbchang.github.io/npe)

## Abstract
We present the Neural Physics Engine (NPE), an object-based neural network
architecture for learning predictive models of intuitive physics. We propose a
factorization of a physical scene into composable object-based representations
and also the NPE architecture whose compositional structure factorizes object
dynamics into pairwise interactions. Our approach draws on the strengths of
both symbolic and neural approaches: like a symbolic physics engine, the NPE is
endowed with generic notions of objects and their interactions, but as a neural
network it can also be trained via stochastic gradient descent to adapt to
specific object properties and dynamics of different worlds. We evaluate the
efficacy of our approach on simple rigid body dynamics in two-dimensional
worlds. By comparing to less structured architectures, we show that our model's
compositional representation of the structure in physical interactions improves
its ability to predict movement, generalize to different numbers of objects,
and infer latent properties of objects such as mass.

Below are some predictions from the model:

<kbd><img src="./demo/balls_n3_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n4_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n5_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n6_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n7_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n8_npe_pred_batch0_ex0.gif" width="125"></kbd>

<kbd><img src="./demo/walls_n2_wO_npe_pred_batch0_ex3.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wL_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wU_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wI_npe_pred_batch0_ex2.gif" width="125"></kbd>

_NOTE: The code in this repository has not been cleaned up._

## Requirements
* [Torch7](http://torch.ch/)
* [Node.js](https://nodejs.org/en/) v6.2.1

### Dependencies
To install lua dependencies, run:

```bash
luarocks install pl
luarocks install torchx
luarocks install nn
luarocks install nngraph
luarocks install rnn
luarocks install gnuplot
luarocks install paths
luarocks install json
```

To install js dependencies, run:
```bash
cd src/js
npm install
```

<!---
Below are python dependencies:
```bash
images2gif==1.0.1
matplotlib==1.4.3
numpy==1.10.4
Pillow==2.8.2
```
-->

## Instructions

<!---
Pretrained network and dataset can be downloaded at: COMING SOON.
--> 

### Generating Data
The code to generate data is adapted from the demo code in
[matter-js](https://github.com/liabru/matter-js).

This is an example of generating 50000 trajectories of 4 balls of variable mass over 60 timesteps. It will create a folder `balls_n4_t60_s50000_m` in the `data/` folder. 
```shell
> cd src/js
> node demo/js/generate.js -e balls -n 4 -t 60 -s 50000 -m
```
This is an example of generating 50000 trajectories of 2 balls over 60 timesteps for wall geometry "U." It will create a folder `walls_n2_t60_s50000_wU` in the `data/` folder.
```shell
> cd src/js
> node demo/js/generate.js -e walls -n 2 -t 60 -s 50000 -w U
```

It takes quite a bit of time to generate 50000 trajectories, so 200
trajectories is enough for debugging purposes. In that case you may want to
change the flags accordingly in the examples below.

### Visualization
Trajectory data is stored in a `.json` file. You can visualize the trajectory by opening `src/js/demo/render.html` in your browser and passing in the `.json` file.


### Training the Model
This is an example of training the model for the `balls_n4_t60_s50000_m` dataset. The model checkpoints are saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`. If you are comfortable looking at code that has not been cleaned up yet, please check out the flags in `src/lua/main.lua`. 
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'balls_n4_t60_ex50000_m'}" -nbrhd -rs -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -fast -lr 0.0003 -model npe -seed 0 -name balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp
```

Here is an example of training on 3, 4, 5 balls of variable mass and testing on 6, 7, 8 balls of variable mass, provided that those datasets have been generated. The model checkpoints are saved in `src/lua/logs/balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m'}" -nbrhd -rs -test_dataset_folders "{'balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m'}" -fast -lr 0.0003 -model npe -seed 0 -name balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp
```

Here is an example of training on "O" and "I" wall geometries and testing on "U" and "I" wall geometries, provided that those datasets have been generated. The model checkpoints are saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL'}" -nbrhd -rs -test_dataset_folders "{'walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -fast -lr 0.0003 -model npe -seed 0 -name walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp 
```

Be sure to look at the command line flags in `main.lua` for more details. You
may want to change the number of training iterations if you are just debugging
. The code defaults to cpu, but you can switch to gpu with the `-cuda` flag.

### Prediction
This is an example of running simulations using trained model that was saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -name balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

This is an example of running simulations using trained model that was saved in `src/lua/logs/balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m','balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m'}" -name balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

This is an example of running simulations using trained model that was saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL','walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -name walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

You can visualize the predictions with `src/js/demo/render.html` and passing in the `.json` files in `src/lua/logs/<experiment_name>/<dataset_name>predictions/<jsonfile>`.

### Inference
This is an example of running mass inference using trained model that was saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -name balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

This is an example of running mass inference using trained model that was saved in `balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m','balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m'}" -name balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

This is an example of running mass inference using trained model that was saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL','walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -name walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

#### Acknowledgements
This project was built with [Torch7](http://torch.ch),
[rnn](https://github.com/Element-Research/rnn), and
[matter-js](http://brm.io/matter-js/). A big thank you to these folks.

We thank Tejas Kulkarni for insightful discussions and guidance. We thank Ilker
Yidirim, Erin Reynolds, Feras Saad, Andreas Stuhlmuller, Adam Lerer, Chelsea
Finn, Jiajun Wu, and the anonymous reviewers for valuable feedback. We thank
Liam Brummit, Kevin Kwok, and Guillermo Webster for help with matter-js. M.
Chang was graciously supported by MIT’s SuperUROP and UROP programs.
