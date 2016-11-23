# [Neural Physics Engine](http://mbchang.github.io/npe)

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

[Project Website](http://mbchang.github.io/npe)

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

_The code in this repository is still under active development, so use at your
own risk._
